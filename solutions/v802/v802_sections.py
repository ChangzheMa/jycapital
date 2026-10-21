"""完整截面的双缓冲加载器；原始行情缓存和目标保持不变。"""

from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

from resident_data import FREQUENCIES, PREVIOUS_DAYS, DAILY_BARS, prepare_device_windows


class SectionBlock:
    def __init__(self, base):
        self.base = base
        self.targets, self.manifest = base.targets, base.manifest
        change = (base.day[1:] != base.day[:-1]) | (base.clock[1:] != base.clock[:-1])
        self.boundaries = np.r_[0, np.flatnonzero(change) + 1, len(base.day)]
        self.max_section_size = int(np.diff(self.boundaries).max())
        if self.max_section_size < 1:
            raise ValueError("完整截面不能是空集合")

    def section(self, number):
        return np.arange(self.boundaries[number], self.boundaries[number + 1])

    def training_targets(self):
        return self.base.training_targets()

    def raw_batch(self, indices, destinations=None):
        return self.base.raw_batch(indices, destinations)


class SectionStore:
    def __init__(self, resident, backbone_artifact=None):
        self.resident = resident
        self.original, self.config, self.protocol = resident.original, resident.config, resident.protocol
        self.segments = resident.segments
        self.block = SectionBlock(resident.block)
        self.backbone_artifact = backbone_artifact

    def months(self, split):
        return ["global"] if split == "train" else list(self.segments[split])

    def load(self, split, segment):
        if split == "train":
            if segment != "global":
                raise ValueError("完整截面训练使用全局数据段")
            return self.block
        return self.resident.load(split, segment)

    def close(self):
        active = getattr(self, "evaluation_cache", None)
        if active is not None:
            active[1].close()
            self.evaluation_cache = None
        self.block, self.resident = None, None


class SectionBatches:
    """一个后台线程和两个可复用 pinned 缓冲；预取不消耗训练随机状态。"""

    def __init__(self, block, order, targets, start, batch_size, device):
        self.block, self.order, self.targets = block, order, targets
        self.start, self.batch_size, self.device = start, block.max_section_size, torch.device(device)

    def __iter__(self):
        cuda = self.device.type == "cuda"
        stream = torch.cuda.Stream(device=self.device) if cuda else None
        slots = []
        lengths = {f: (PREVIOUS_DAYS[f] + int(f == "bar1m")) * DAILY_BARS[f] for f in FREQUENCIES}
        for _ in range(2):
            tensors = {f: torch.empty(self.batch_size, length, 72, dtype=torch.float16, pin_memory=cuda)
                       for f, length in lengths.items()}
            slots.append({"tensors": tensors, "numpy": {f: v.numpy() for f, v in tensors.items()}, "event": None})
        offsets = iter(range(self.start, len(self.order)))

        def fill(offset, slot):
            if slot["event"] is not None:
                slot["event"].synchronize()
            selected = self.block.section(self.order[offset])
            raw, repairs, clock = self.block.raw_batch(selected, slot["numpy"])
            return offset, selected, slot, raw, repairs, clock

        def transfer(packed):
            offset, selected, slot, raw, repairs, clock = packed
            def copy_and_prepare():
                values = {f: slot["tensors"][f][:len(selected)].to(self.device, non_blocking=cuda) for f in FREQUENCIES}
                fixed = {f: tuple(torch.from_numpy(a).to(self.device, non_blocking=cuda) for a in pair)
                         for f, pair in repairs.items()}
                clock_tensor = torch.from_numpy(clock).to(self.device, non_blocking=cuda)
                target = torch.from_numpy(self.targets[selected]).to(self.device, non_blocking=cuda)
                windows = prepare_device_windows(values, fixed, clock_tensor)
                return windows, clock_tensor, target
            if cuda:
                with torch.cuda.stream(stream):
                    values = copy_and_prepare()
                    ready = torch.cuda.Event()
                    ready.record(stream)
                slot["event"] = ready
            else:
                values, ready = copy_and_prepare(), None
            # 保留所有主机端数组，直到对应批次被消费，覆盖异步复制的生命周期。
            return offset, selected, values, ready, packed

        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                first = next(offsets, None)
                if first is None:
                    return
                current = transfer(pool.submit(fill, first, slots[0]).result())
                number = 1
                offset = next(offsets, None)
                pending = pool.submit(fill, offset, slots[1]) if offset is not None else None
                while current is not None:
                    upcoming = transfer(pending.result()) if pending is not None else None
                    number += 1
                    offset = next(offsets, None)
                    pending = pool.submit(fill, offset, slots[number % 2]) if offset is not None else None
                    position, selected, (windows, clock, target), ready, host_data = current
                    if cuda:
                        compute = torch.cuda.current_stream(self.device)
                        compute.wait_event(ready)
                        for value in (*windows.values(), clock, target):
                            value.record_stream(compute)
                    yield position, selected, windows, clock, target
                    current = upcoming
            finally:
                if stream is not None:
                    stream.synchronize()
