"""完整训练日期的股票日分组与双缓冲传输；不改写底层行情和标签。"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import torch
import numpy as np

from artifacts import read_json, sha256, write_json
from v1501_resident import FREQUENCIES, PREVIOUS_DAYS, DAILY_BARS, prepare_device_windows


class CalendarDayPlan:
    def __init__(self, view, config, protocol):
        rows = view.index.representatives
        if hasattr(view.base, 'arrays'):
            dates = np.asarray(view.base.arrays['days'])[view.base.day[rows]].astype('datetime64[ns]').astype('datetime64[D]')
        else:
            dates = view.base.keys.iloc[rows].date.to_numpy(dtype='datetime64[ns]').astype('datetime64[D]')
        interval = protocol['splits']['train']
        if (np.isnat(dates).any() or (dates < np.datetime64(interval['start'], 'D')).any()
                or (dates >= np.datetime64(interval['stop'], 'D')).any()):
            raise ValueError('日期分组只能包含固定训练区间')
        supervised_dates = np.unique(dates[view.supervised])
        view.eligible &= np.isin(dates,supervised_dates)
        eligible = np.flatnonzero(view.eligible)
        if not len(eligible):
            raise ValueError('完整日期训练没有有效股票日')
        ordered = eligible[np.argsort(dates[eligible], kind='stable')]
        self.dates, starts, counts = np.unique(dates[ordered], return_index=True, return_counts=True)
        self.groups = [ordered[start:start + count] for start, count in zip(starts, counts)]
        self.maximum = int(counts.max())
        if self.maximum > config['training']['stock_day_batch_size']:
            raise ValueError('完整交易日超过已预检的批量上限，不能拆成不完整截面')
        raw_weights = np.asarray(view.sample_weights, dtype=np.float64)
        date_weights = raw_weights[ordered[starts]]
        for groups, weight in zip(self.groups, date_weights):
            if not np.all(raw_weights[groups] == weight):
                raise ValueError('同日股票的时间权重不一致')
        normalizer = float(date_weights.mean())
        view.sample_weights = raw_weights / normalizer
        view.temporal_weighting_record = dict(view.temporal_weighting_record,
            raw_weights_sha256=view.temporal_weighting_record['weights_sha256'],
            weights_sha256=hashlib.sha256(view.sample_weights.tobytes()).hexdigest(),
            normalization='mean_raw_weight_over_all_eligible_training_dates', normalizer=normalizer,
            minimum=float(view.sample_weights.min()), maximum=float(view.sample_weights.max()),
            mean=float(view.sample_weights.mean()))
        self.record = {
            'format': 'jycapital-complete-calendar-training-v1', 'cursor_unit': 'calendar_day',
            'dates': len(self.dates), 'context_stock_days': len(eligible), 'supervised_stock_days': int(view.supervised.sum()),
            'minimum_stock_days': int(counts.min()), 'maximum_stock_days': self.maximum,
            'dates_sha256': hashlib.sha256(self.dates.tobytes()).hexdigest(),
            'ordered_stock_days_sha256': hashlib.sha256(ordered.tobytes()).hexdigest(),
            'day_sizes_sha256': hashlib.sha256(counts.astype(np.int64).tobytes()).hexdigest(),
            'temporal_normalizer': normalizer, 'normalized_date_weight_mean': float((date_weights / normalizer).mean()),
            'microbatch_stock_days': config['training']['microbatch_stock_days'],
        }

    def order(self, seed, epoch):
        order = np.arange(len(self.groups), dtype=np.int64)
        np.random.default_rng(np.random.SeedSequence([seed, epoch, 1301])).shuffle(order)
        return order

    def record_identity(self, run):
        path = Path(run) / 'calendar-day-training.json'
        record = dict(self.record, data_manifest_sha256=sha256(Path(run) / 'data-manifest.json'))
        if path.exists() and read_json(path) != record:
            raise ValueError('恢复时完整日期成员、时间权重或计算块改变')
        write_json(path, record)


def calendar_batches(view, plan, order, start, device):
    """按完整日边界交付；合成普通视图也遵守同一日期顺序。"""
    device = torch.device(device)
    if hasattr(view, 'raw_batch'):
        yield from ResidentCalendarBatches(view, plan, order, start, device)
        return
    for offset in range(start, len(order)):
        selected = plan.groups[order[offset]]
        windows, clock = view.batch(selected)
        yield (offset, selected, {f: torch.from_numpy(x).to(device) for f, x in windows.items()},
               torch.from_numpy(clock).to(device), torch.from_numpy(view.targets[selected]).to(device))


class ResidentCalendarBatches:
    """一个后台线程和两个可复用 pinned 缓冲；预取不消耗训练随机状态。"""

    def __init__(self, view, plan, order, start, device):
        self.block, self.plan, self.order, self.targets = view, plan, order, view.targets
        self.start, self.batch_size, self.device = start, plan.maximum, torch.device(device)

    def __iter__(self):
        cuda = self.device.type == "cuda"
        stream = torch.cuda.Stream(device=self.device) if cuda else None
        slots = []
        lengths = {f: (PREVIOUS_DAYS[f] + 1) * DAILY_BARS[f] for f in FREQUENCIES}
        for _ in range(2):
            tensors = {f: torch.empty(self.batch_size, length, 72, dtype=torch.float16, pin_memory=cuda)
                       for f, length in lengths.items()}
            slots.append({"tensors": tensors, "numpy": {f: v.numpy() for f, v in tensors.items()}, "event": None})
        offsets = iter(range(self.start, len(self.order)))

        def fill(offset, slot):
            if slot["event"] is not None:
                slot["event"].synchronize()
            selected = self.plan.groups[self.order[offset]]
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

