"""三频训练数据的只读 FP16 常驻缓存、全局采样与双缓冲传输。"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import time

# Windows 下先加载 CUDA DLL，再加载 NumPy/Arrow。
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil

from artifacts import read_json, sha256, write_json

FORMAT = "jycapital-resident-fp16-v1"
PRECISION = "fp16_with_fp32_overflow"
FREQUENCIES = ("bar1m", "bar5m", "bar30m")
DAILY_BARS = {"bar1m": 240, "bar5m": 48, "bar30m": 8}
PREVIOUS_DAYS = {"bar1m": 1, "bar5m": 5, "bar30m": 20}
CLOCK_MINUTES = np.array([570, 600, 630, 660, 690, 810, 840, 870], np.int64)
MINUTE_NS = 60_000_000_000
DAY_NS = 1440 * MINUTE_NS
VALUE_COLUMNS = [f"{kind}_{field}" for kind in ("z", "c") for field in (
    ["open", "high", "low", "close", "volume", "amount"] +
    [f"{prefix}{level}" for prefix in ("ask_price", "bid_price", "ask_volume", "bid_volume",
                                      "ask_num_orders", "bid_num_orders") for level in range(1, 6)])]


def validate_resident_config(config):
    spec = config["training"]
    mode = spec.get("batching")
    if mode == "monthly_shuffled_samples":
        if config["data"].get("input_precision", "float32") != "float32":
            raise ValueError("月内对照模式要求原始 FP32 输入")
        return
    if mode != "global_shuffled_samples" or config["data"].get("input_precision") != PRECISION:
        raise ValueError("全局随机训练要求 FP16 舍入并保留 FP32 溢出值")
    cache = spec.get("resident_cache", {})
    if cache.get("format") != FORMAT or cache.get("prefetch_batches") != 2:
        raise ValueError("常驻缓存格式或双缓冲配置不匹配")
    for key in ("build_batch_rows", "evaluation_days"):
        value = cache.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(key + " 必须为正整数")
    reserve = cache.get("memory_reserve_gib")
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("memory_reserve_gib 必须为有限非负数")


def require_memory(array_bytes, reserve_gib):
    available = psutil.virtual_memory().available
    required = int(array_bytes + reserve_gib * 2**30)
    if available < required:
        raise MemoryError(f"全量缓存需要 {array_bytes / 2**30:.2f} GiB，另预留 {reserve_gib:.2f} GiB；"
                          f"当前可用 {available / 2**30:.2f} GiB。请先释放内存或使用月内对照配置。")


class TrainingPlan:
    """只读股票池索引，计算所有样本需要的股票／交易日并集。"""

    def __init__(self, store):
        interval = store.protocol["splits"]["train"]
        pool = store.universe
        self.keys = pool.loc[(pool.date >= interval["start"]) & (pool.date < interval["stop"]),
                             ["date", "instrument"]].sort_values(["date", "instrument"]).reset_index(drop=True)
        if self.keys.empty or self.keys.duplicated(["date", "instrument"]).any() or self.keys.isna().any().any():
            raise ValueError("训练股票池为空、含空键或重复键")
        times = self.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.days = np.unique(times // DAY_NS * DAY_NS)
        self.stocks = pd.Index(sorted(self.keys.instrument.unique()))
        self.stock = self.stocks.get_indexer(self.keys.instrument).astype(np.int32)
        self.day = np.searchsorted(self.days, times // DAY_NS * DAY_NS).astype(np.int32)
        minutes = (times // MINUTE_NS) % 1440
        self.clock = np.searchsorted(CLOCK_MINUTES, minutes)
        if (times % MINUTE_NS).any() or np.any(self.clock >= 8) or not np.array_equal(CLOCK_MINUTES[self.clock], minutes):
            raise ValueError("训练键包含非法预测时刻")
        self.clock = self.clock.astype(np.uint8)
        present = np.zeros((len(self.days), len(self.stocks)), bool)
        present[self.day, self.stock] = True
        self.maps, self.counts = {}, {}
        for frequency in FREQUENCIES:
            needed = present.copy() if frequency == "bar1m" else np.zeros_like(present)
            for offset in range(1, PREVIOUS_DAYS[frequency] + 1):
                needed[:-offset] |= present[offset:]
            # 同一股票的各交易日相邻保存。0 号块统一表示缺少历史或缺少股票数据。
            stocks, days = np.nonzero(needed.T)
            mapping = np.zeros(present.shape, np.int32)
            mapping[days, stocks] = np.arange(1, len(days) + 1, dtype=np.int32)
            self.maps[frequency] = mapping
            self.counts[frequency] = len(days) + 1

    def estimate(self):
        features = {f: self.counts[f] * DAILY_BARS[f] * 72 * 2 for f in FREQUENCIES}
        metadata = sum(v.nbytes for v in self.maps.values()) + self.days.nbytes
        metadata += self.stock.nbytes + self.day.nbytes + self.clock.nbytes + len(self.keys) * 16
        # 每块的例外值起止索引；另外预留一个全局 int64 排列。
        metadata += sum((count + 1) * 8 for count in self.counts.values()) + len(self.keys) * 8
        return {"samples": len(self.keys), "days": len(self.days), "stocks": len(self.stocks),
                "feature_bytes": features, "metadata_and_order_bytes": metadata,
                "resident_bytes_before_exceptions": sum(features.values()) + metadata}


def cache_key(store, data_manifest, transform):
    module = inspect.getmodule(transform)
    signature = (inspect.getsource(transform) + inspect.getsource(module.standardized_targets)
                 + inspect.getsource(module.section_indices))
    identity = {"format": FORMAT, "builder_sha256": sha256(__file__), "protocol": store.protocol,
                "data": store.config["data"], "sources": data_manifest,
                "target_transform_sha256": hashlib.sha256(signature.encode("utf-8")).hexdigest()}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_array(directory, name, value):
    np.save(directory / (name + ".npy"), value, allow_pickle=False)


def build_cache(store, directory, key, transform, *, plan=None, report=None):
    """分批转换既有 z/c 文件；不重新拟合统计，不展开样本窗口。"""
    directory = Path(directory)
    if directory.exists():
        saved = read_json(directory / "manifest.json")
        if saved.get("status") != "complete" or saved.get("cache_key") != key or saved.get("format") != FORMAT:
            raise ValueError("已有缓存未完成或来源不匹配，请使用新的缓存目录")
        return saved
    plan = plan or TrainingPlan(store)
    ancestor = next(path for path in directory.parents if path.exists())
    required_disk = plan.estimate()["resident_bytes_before_exceptions"] + 1024**3
    if shutil.disk_usage(ancestor).free < required_disk:
        raise OSError("磁盘空间不足以保存完整常驻缓存及构建余量")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"format": FORMAT, "cache_key": key, "status": "building", "estimate": plan.estimate(),
                "input_precision": PRECISION, "stocks": plan.stocks.tolist(), "frequencies": {}}
    write_json(directory / "manifest.json", manifest)
    report = report or (lambda *args, **kwargs: None)
    try:
        interval = store.protocol["splits"]["train"]
        labels = pq.read_table(store.data_root / (store.protocol["label_table"] + ".parquet"),
                               columns=["date", "instrument", store.protocol["label_column"]],
                               filters=[("date", ">=", pd.Timestamp(interval["start"])),
                                        ("date", "<", pd.Timestamp(interval["stop"]))]).to_pandas()
        keys = plan.keys.merge(labels, on=["date", "instrument"], how="left", validate="one_to_one")
        del labels
        horizon = keys.date + pd.Timedelta(minutes=store.protocol["label_horizon_minutes"])
        admissible = (horizon < pd.Timestamp(interval["stop"])) & (horizon.dt.normalize() == keys.date.dt.normalize())
        keys.loc[~admissible, store.protocol["label_column"]] = np.nan
        targets, skipped = transform(keys, store.protocol)
        if not np.isfinite(targets).any():
            raise ValueError("训练期没有可优化的标准化标签")
        raw_labels = keys[store.protocol["label_column"]].to_numpy(dtype=np.float64)
        catalog = {"sections": int(keys.date.nunique()), "rows": len(keys),
                   "finite_label_rows": int(np.isfinite(raw_labels).sum()), "skipped": {}}
        for stamp, reason in skipped.items():
            positions = keys.date == pd.Timestamp(stamp)
            catalog["skipped"][stamp] = {"reason": reason, "rows": int(positions.sum()),
                                         "finite_labels": int(np.isfinite(raw_labels[positions]).sum())}
        manifest.update(catalog=catalog, eligible_samples=int(np.isfinite(targets).sum()))
        for name, value in (("days", plan.days), ("sample_stock", plan.stock), ("sample_day", plan.day),
                            ("sample_clock", plan.clock), ("targets", targets)):
            _save_array(directory, name, value)
        del keys, raw_labels, targets, horizon, admissible
        for frequency in FREQUENCIES:
            bars = DAILY_BARS[frequency]
            mapping = plan.maps[frequency]
            _save_array(directory, frequency + "-map", mapping)
            path = directory / (frequency + "-values.npy")
            values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                               shape=(plan.counts[frequency], bars, 72))
            values[:] = np.nan
            step = int(frequency[3:-1])
            slot_minutes = np.r_[np.arange(570 + step, 691, step), np.arange(780 + step, 901, step)]
            slot_lookup = np.full(1440, -1, np.int16)
            slot_lookup[slot_minutes] = np.arange(bars)
            indices, exceptions = [], []
            count = overflow_count = 0
            max_abs_error = 0.0
            for month in store.months("train"):
                source = store.path(frequency, month)
                if source is None or not source.is_file():
                    raise FileNotFoundError(source)
                parquet = pq.ParquetFile(source)
                for batch in parquet.iter_batches(batch_size=store.config["training"]["resident_cache"]["build_batch_rows"],
                                                  columns=["date", "instrument", *VALUE_COLUMNS]):
                    frame = batch.to_pandas()
                    times = frame.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                    day = np.searchsorted(plan.days, times // DAY_NS * DAY_NS)
                    stock = plan.stocks.get_indexer(frame.instrument)
                    slot = slot_lookup[(times // MINUTE_NS) % 1440]
                    good = (day < len(plan.days)) & (stock >= 0) & (slot >= 0) & (times % MINUTE_NS == 0)
                    good &= plan.days[day.clip(0, len(plan.days) - 1)] == times // DAY_NS * DAY_NS
                    chosen = np.flatnonzero(good)
                    block = mapping[day[chosen], stock[chosen]]
                    chosen, block = chosen[block != 0], block[block != 0]
                    if not len(chosen):
                        continue
                    raw = frame.iloc[chosen][VALUE_COLUMNS].to_numpy(dtype=np.float32)
                    with np.errstate(over="ignore", invalid="ignore"):
                        half = raw.astype(np.float16)
                    finite = np.isfinite(raw)
                    overflow = finite & ~np.isfinite(half)
                    row, column = np.nonzero(overflow)
                    if len(row):
                        indices.append(((block[row].astype(np.int64) * bars + slot[chosen[row]]) * 72 + column))
                        exceptions.append(raw[row, column])
                        half[overflow] = 0
                        overflow_count += len(row)
                        # 例外本应稀疏；在拼接／排序之前检查额外内存，避免大量例外导致失控。
                        require_memory(overflow_count * 32, 0.5)
                    normal = finite & ~overflow
                    if normal.any():
                        max_abs_error = max(max_abs_error, float(np.abs(half[normal].astype(np.float32) - raw[normal]).max()))
                    values[block, slot[chosen]] = half
                    count += int(finite.sum())
                report("resident_cache_partition", frequency=frequency, month=month)
            values.flush()
            del values
            flat = np.concatenate(indices) if indices else np.empty(0, np.int64)
            restored = np.concatenate(exceptions) if exceptions else np.empty(0, np.float32)
            order = np.argsort(flat)
            flat, restored = flat[order], restored[order]
            if len(flat) > 1 and np.any(np.diff(flat) <= 0):
                raise ValueError("预处理文件包含重复的溢出值位置")
            block = flat // (bars * 72)
            pointers = np.searchsorted(block, np.arange(plan.counts[frequency] + 1)).astype(np.int64)
            _save_array(directory, frequency + "-exception-pointers", pointers)
            _save_array(directory, frequency + "-exception-offsets", (flat % (bars * 72)).astype(np.int32))
            _save_array(directory, frequency + "-exception-values", restored)
            manifest["frequencies"][frequency] = {"blocks": plan.counts[frequency], "bars": bars,
                "finite_values": count, "fp32_exceptions": overflow_count, "max_absolute_rounding_error": max_abs_error}
            report("resident_cache_frequency", frequency=frequency, **manifest["frequencies"][frequency])
        manifest["files"] = {}
        for path in sorted(directory.glob("*.npy")):
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path),
                                            "shape": list(mapped.shape), "dtype": mapped.dtype.str,
                                            "array_bytes": mapped.nbytes}
            del mapped
        manifest["resident_array_bytes"] = sum(v["array_bytes"] for v in manifest["files"].values())
        manifest["status"] = "complete"
        write_json(directory / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        write_json(directory / "manifest.json", manifest)
        raise


def _load_resident_array(path, record):
    """直接读入最终内存，读取时校验哈希，不保留完整 mmap 或第二份数组。"""
    if path.stat().st_size != record["bytes"]:
        raise ValueError("缓存文件大小已改变: " + path.name)
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        if dtype.hasobject or fortran or list(shape) != record["shape"] or dtype.str != record["dtype"]:
            raise ValueError("缓存数组元数据不匹配: " + path.name)
        header_size = stream.tell()
        stream.seek(0)
        digest = hashlib.sha256(stream.read(header_size))
        value = np.empty(shape, dtype=dtype)
        buffer = memoryview(value).cast("B")
        position = 0
        while position < len(buffer):
            end = min(position + 16 * 1024**2, len(buffer))
            size = stream.readinto(buffer[position:end])
            if not size:
                raise ValueError("缓存文件被截断: " + path.name)
            digest.update(buffer[position:position + size])
            position += size
        if stream.read(1) or digest.hexdigest() != record["sha256"]:
            raise ValueError("缓存文件校验失败: " + path.name)
    value.flags.writeable = False
    return value


class ResidentBlock:
    """训练期数据全部驻留；batch 只按预计算块号进行 gather。"""

    def __init__(self, directory, key, reserve_gib=8.0, report=None):
        directory = Path(directory)
        self.manifest = read_json(directory / "manifest.json")
        m = self.manifest
        if m.get("format") != FORMAT or m.get("status") != "complete" or m.get("cache_key") != key:
            raise ValueError("不能加载未完成或来源不同的常驻缓存")
        require_memory(m["resident_array_bytes"] + m["catalog"]["rows"] * 8, reserve_gib)
        arrays = {}
        for name, record in m["files"].items():
            path = directory / name
            if path.parent != directory or path.suffix != ".npy":
                raise ValueError("非法缓存文件名")
            arrays[name[:-4]] = _load_resident_array(path, record)
            if report:
                report("resident_array_loaded", array=name, bytes=record["array_bytes"])
        self.arrays = arrays
        self.targets = arrays["targets"]
        self.clock = arrays["sample_clock"]
        self.day, self.stock = arrays["sample_day"], arrays["sample_stock"]
        self.day_plans = {}
        for f in FREQUENCIES:
            offsets = np.arange(-PREVIOUS_DAYS[f], 1 if f == "bar1m" else 0)
            self.day_plans[f] = np.arange(len(arrays["days"]))[:, None] + offsets

    def training_targets(self):
        return self.targets, {stamp: row["reason"] for stamp, row in self.manifest["catalog"]["skipped"].items()}

    def raw_batch(self, indices, destinations=None):
        indices = np.asarray(indices)
        count = len(indices)
        raw, repairs = {}, {}
        for f in FREQUENCIES:
            days = self.day_plans[f][self.day[indices]]
            block = self.arrays[f + "-map"][days.clip(0), self.stock[indices, None]]
            block = np.where(days >= 0, block, 0)
            length = days.shape[1] * DAILY_BARS[f]
            output = (destinations[f][:count] if destinations is not None else
                      np.empty((count, length, 72), np.float16))
            np.take(self.arrays[f + "-values"], block.reshape(-1), axis=0,
                    out=output.reshape(-1, DAILY_BARS[f], 72), mode="clip")
            pointers = self.arrays[f + "-exception-pointers"]
            starts, ends = pointers[block], pointers[block + 1]
            rows, slots = np.nonzero(ends > starts)
            positions, values = [], []
            for row, slot in zip(rows, slots):
                start, stop = starts[row, slot], ends[row, slot]
                local = self.arrays[f + "-exception-offsets"][start:stop]
                positions.append(row * length * 72 + slot * DAILY_BARS[f] * 72 + local.astype(np.int64))
                values.append(self.arrays[f + "-exception-values"][start:stop])
            repairs[f] = (np.concatenate(positions) if positions else np.empty(0, np.int64),
                          np.concatenate(values) if values else np.empty(0, np.float32))
            raw[f] = output
        return raw, repairs, np.eye(8, dtype=np.float32)[self.clock[indices]]

    def batch(self, indices):
        """CPU 参考路径，用于因果性及普通切窗对照。"""
        raw, repairs, clock = self.raw_batch(indices)
        tensors = {f: torch.from_numpy(v) for f, v in raw.items()}
        repair_tensors = {f: tuple(torch.from_numpy(x) for x in pair) for f, pair in repairs.items()}
        windows = prepare_device_windows(tensors, repair_tensors, torch.from_numpy(clock))
        return {f: x.numpy() for f, x in windows.items()}, clock


def prepare_device_windows(raw, repairs, clock):
    """FP16 传输后恢复 FP32、稀疏极值与原有 75 通道语义。"""
    result = {}
    for f in FREQUENCIES:
        x = raw[f].float()
        indices, values = repairs[f]
        if indices.numel():
            x.reshape(-1).index_copy_(0, indices, values)
        length = x.shape[1]
        available = (torch.arange(length, device=x.device)[None, :] <
                     (240 + clock.argmax(dim=1) * 30)[:, None]) if f == "bar1m" else torch.ones(
                         (len(x), length), dtype=torch.bool, device=x.device)
        finite = torch.isfinite(x) & available[:, :, None]
        non_book = torch.cat([finite[:, :, :6], finite[:, :, 36:42]], dim=2).all(dim=2)
        book = torch.cat([finite[:, :, 6:36], finite[:, :, 42:72]], dim=2).all(dim=2)
        flags = torch.stack([non_book, book, finite.any(dim=2)], dim=2).to(torch.float32)
        result[f] = torch.cat([torch.where(finite, x, 0), flags], dim=2).transpose(1, 2)
    return result


class ResidentBatches:
    """一个后台线程和两个可复用 pinned 缓冲；预取不消耗训练随机状态。"""

    def __init__(self, block, order, targets, start, batch_size, device):
        self.block, self.order, self.targets = block, order, targets
        self.start, self.batch_size, self.device = start, batch_size, torch.device(device)

    def __iter__(self):
        cuda = self.device.type == "cuda"
        stream = torch.cuda.Stream(device=self.device) if cuda else None
        slots = []
        lengths = {f: (PREVIOUS_DAYS[f] + int(f == "bar1m")) * DAILY_BARS[f] for f in FREQUENCIES}
        for _ in range(2):
            tensors = {f: torch.empty(self.batch_size, length, 72, dtype=torch.float16, pin_memory=cuda)
                       for f, length in lengths.items()}
            slots.append({"tensors": tensors, "numpy": {f: v.numpy() for f, v in tensors.items()}, "event": None})
        offsets = iter(range(self.start, len(self.order), self.batch_size))

        def fill(offset, slot):
            if slot["event"] is not None:
                slot["event"].synchronize()
            selected = self.order[offset:offset + self.batch_size]
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


class ResidentStore:
    """训练为一个全局数据段，验证／测试按有限交易日块加载。"""

    def __init__(self, original, block, evaluation_days):
        self.original, self.block = original, block
        self.protocol, self.config = original.protocol, original.config
        self.segments = {}
        for split in ("validation", "test"):
            interval = self.protocol["splits"][split]
            days = original.days[(original.days >= np.datetime64(interval["start"])) &
                                 (original.days < np.datetime64(interval["stop"]))]
            self.segments[split] = {}
            for start in range(0, len(days), evaluation_days):
                chunk = days[start:start + evaluation_days]
                lo, hi = pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1]) + pd.Timedelta(days=1)
                self.segments[split][str(lo.date())] = (lo, hi)

    def months(self, split):
        return ["global"] if split == "train" else list(self.segments[split])

    def load(self, split, segment):
        if split == "train":
            if segment != "global":
                raise ValueError("全局训练段标识错误")
            return self.block
        return self.original.load_range(split, *self.segments[split][segment])


def open_resident_store(store, run, transform, directory=None, report=None):
    key = cache_key(store, read_json(Path(run) / "data-manifest.json"), transform)
    directory = Path(directory) if directory else store.training_run.parent / "resident-cache" / key
    spec = store.config["training"]["resident_cache"]
    started = time.perf_counter()
    if not directory.exists():
        plan = TrainingPlan(store)
        require_memory(plan.estimate()["resident_bytes_before_exceptions"], spec["memory_reserve_gib"])
        build_cache(store, directory, key, transform, plan=plan, report=report)
        del plan
    block = ResidentBlock(directory, key, spec["memory_reserve_gib"], report)
    record = {"cache_key": key, "format": FORMAT, "path": str(directory.resolve()),
              "manifest_sha256": sha256(directory / "manifest.json"),
              "resident_array_bytes": block.manifest["resident_array_bytes"],
              "input_precision": PRECISION, "load_and_build_seconds": time.perf_counter() - started}
    previous = Path(run) / "resident-cache.json"
    if previous.exists() and read_json(previous)["cache_key"] != key:
        raise ValueError("断点恢复的缓存来源不同")
    write_json(previous, record)
    return ResidentStore(store, block, spec["evaluation_days"])
