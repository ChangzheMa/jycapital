"""盘前1分钟旁路缓存、因果取窗与异步输入准备；不读取收益标签。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from artifacts import read_json, write_json, sha256

FORMAT = "jycapital-preopen-context-fp32-v1"
PREOPEN_KEY = "preopen1m"
KEYS = ["date", "instrument"]
DAY_NS = 86400000000000
MINUTE_NS = 60000000000
MINUTES = np.arange(556, 566, dtype=np.int64)
BARS = len(MINUTES)
FIELDS = ["open", "high", "low", "close", "volume", "amount"] + [
    f"{prefix}{level}" for prefix in ("ask_price", "bid_price", "ask_volume", "bid_volume",
                                     "ask_num_orders", "bid_num_orders") for level in range(1, 6)]
VALUE_COLUMNS = [kind + "_" + field for kind in ("z", "c") for field in FIELDS]


def validate_preopen_config(config):
    spec = config["preopen_context"]
    if config["data"]["normalization"] != "frozen_v2_z_and_c":
        raise ValueError("本盘前输入契约要求冻结 z/c 字段")
    expected = {"version": "preopen_phase_v1", "frequency": "bar1m", "daily_bars": BARS,
                "minutes": MINUTES.tolist(), "include_today": True,
                "input_channels": 75, "input_precision": "follow_short_input_precision",
                "cache_storage": "float32", "cache_loading": "read_only_mmap",
                "residual_output_initialization": "zero", "no_history": "zero_residual",
                "sharing": "one_encoding_per_stock_day", "read_memory": "previous_daily_states_and_current_bars"}
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("盘前历史输入契约不匹配")
    if isinstance(spec.get("previous_days"), bool) or not isinstance(spec.get("previous_days"), int) or spec["previous_days"] < 0:
        raise ValueError("盘前历史交易日数必须为非负整数")
    for key in ("day_dim", "attention_heads", "intraday_layers", "encoder_layers", "ff_multiplier"):
        if isinstance(spec.get(key), bool) or not isinstance(spec.get(key), int) or spec[key] < 1:
            raise ValueError("盘前分支配置必须为正整数: " + key)
    if spec["day_dim"] % spec["attention_heads"]:
        raise ValueError("盘前隐藏宽度必须整除注意力头数")
    if not 0 <= spec["dropout"] < 1 or not 0 < spec["residual_scale_initial"] <= 1:
        raise ValueError("盘前分支 dropout 或残差尺度非法")


def prepare_values(values, precision):
    """[股票日,日期,10,72] -> [股票日,75,日期*10]，精度与短期路径一致。"""
    values = np.asarray(values, dtype=np.float32)
    length = int(np.prod(values.shape[1:-1]))
    values = values.reshape(len(values), length, 72)
    if precision == "fp16_with_fp32_overflow":
        with np.errstate(over="ignore", invalid="ignore"):
            rounded = values.astype(np.float16).astype(np.float32)
        overflow = np.isfinite(values) & ~np.isfinite(rounded)
        np.copyto(rounded, values, where=overflow)
        values = rounded
    elif precision != "float32":
        raise ValueError("未知盘前输入精度")
    finite = np.isfinite(values)
    result = np.zeros((len(values), 75, values.shape[1]), np.float32)
    result[:, :72] = np.where(finite, values, 0).transpose(0, 2, 1)
    result[:, 72] = np.concatenate([finite[:, :, :6], finite[:, :, 36:42]], axis=2).all(axis=2)
    result[:, 73] = np.concatenate([finite[:, :, 6:36], finite[:, :, 42:72]], axis=2).all(axis=2)
    result[:, 74] = finite.any(axis=2)
    return result


def history_positions(calendar, dates, horizon):
    calendar = np.asarray(calendar, dtype="datetime64[ns]").astype(np.int64)
    dates = np.asarray(dates, dtype="datetime64[D]").astype("datetime64[ns]").astype(np.int64)
    if not len(calendar) or np.any(np.diff(calendar) <= 0):
        raise ValueError("交易日历必须非空且严格递增")
    positions = np.searchsorted(calendar, dates)
    if np.any(positions >= len(calendar)) or not np.array_equal(calendar[positions], dates):
        raise ValueError("预测日期不在历史上下文日历中")
    return positions[:, None] + np.arange(-horizon, 1, dtype=np.int64)[None, :]


def from_grid(grid, stocks, calendar, keys, config):
    """只取当前及此前交易日的09:16—09:25槽位，均早于已声明的首个预测时点。"""
    times, values = grid
    positions = history_positions(calendar, keys.date.to_numpy(), config["preopen_context"]["previous_days"])
    calendar = np.asarray(calendar, dtype="datetime64[ns]").astype(np.int64)
    wanted = calendar[positions.clip(0)][:, :, None] + MINUTES[None, None, :] * MINUTE_NS
    slots = np.searchsorted(times, wanted)
    valid = (positions >= 0)[:, :, None] & (slots < len(times))
    if len(times):
        valid &= times[slots.clip(0, len(times) - 1)] == wanted
    stock = pd.Index(stocks).get_indexer(keys.instrument)
    if (stock < 0).any():
        raise ValueError("盘前网格缺少请求股票")
    raw = np.full((*wanted.shape, 72), np.nan, np.float32)
    if len(times):
        gathered = values[slots.clip(0, len(times) - 1), stock[:, None, None]]
        raw = np.where(valid[..., None], gathered, np.nan)
    return prepare_values(raw, config["data"]["input_precision"])


def input_identity(store, split):
    # 只绑定输入语义，不把网络宽度、损失或模型名称放入数据缓存键。
    paths = [store.training_run / "manifest.json", store.holdout_run / "manifest.json"]
    sources = [sha256(path) for path in paths]
    records = {}
    for name, prepared in (("train", store.training_run), ("holdout", store.holdout_run)):
        if read_json(prepared / "manifest.json").get("status") != "complete":
            raise ValueError("盘前输入要求已完成的预处理来源")
        for path in sorted((prepared / "records").rglob("*.json")):
            if path.parent.name == "bar1m":
                record = read_json(path)
                records[name + "/" + path.relative_to(prepared / "records").as_posix()] = {
                    "bytes": record["bytes"], "sha256": record["sha256"]}
    identity = {"format": FORMAT, "split": split, "protocol": store.protocol,
                "horizon": store.config["preopen_context"]["previous_days"],
                "normalization": store.config["data"]["normalization"],
                "preprocessing_manifests": sources, "bar1m_records": records,
                "universe_sha256": sha256(store.data_root / (store.protocol["universe_table"] + ".parquet")),
                "builder": sha256(__file__)}
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), identity


def build_cache(store, split, directory, key, identity):
    horizon = store.config["preopen_context"]["previous_days"]
    interval = store.protocol["splits"][split]
    keys = store.universe.loc[(store.universe.date >= interval["start"]) &
                              (store.universe.date < interval["stop"]), KEYS]
    if keys.empty or keys.duplicated(KEYS).any() or keys[KEYS].isna().any().any():
        raise ValueError("盘前缓存预测键为空或重复")
    all_days = np.asarray(store.days, dtype="datetime64[ns]").astype(np.int64)
    target_days = keys.date.to_numpy(dtype="datetime64[D]").astype("datetime64[ns]").astype(np.int64)
    lo = max(0, int(np.searchsorted(all_days, target_days.min())) - horizon)
    hi = int(np.searchsorted(all_days, target_days.max())) + 1
    days = all_days[lo:hi]
    stocks = pd.Index(sorted(keys.instrument.unique()))
    present = np.zeros((len(days), len(stocks)), bool)
    present[np.searchsorted(days, target_days), stocks.get_indexer(keys.instrument)] = True
    prefix = np.vstack([np.zeros((1, len(stocks)), np.int32), np.cumsum(present, axis=0, dtype=np.int32)])
    indices = np.arange(len(days))
    needed = prefix[np.minimum(indices + horizon + 1, len(days))] > prefix[indices]
    stock_rows, day_rows = np.nonzero(needed.T)
    mapping = np.zeros(present.shape, np.int32)
    mapping[day_rows, stock_rows] = np.arange(1, len(day_rows) + 1, dtype=np.int32)
    count = len(day_rows) + 1
    directory.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory.parent).free < count * BARS * 72 * 4 + 1024 ** 3:
        raise OSError("磁盘空间不足以构建盘前旁路缓存")
    with tempfile.TemporaryDirectory(prefix=key[:12] + "-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        np.save(staging / "days.npy", days, allow_pickle=False)
        np.save(staging / "map.npy", mapping, allow_pickle=False)
        values = np.lib.format.open_memmap(staging / "values.npy", mode="w+", dtype=np.float32,
                                           shape=(count, BARS, 72))
        values[:] = np.nan
        occupied = np.zeros((count, BARS), bool)
        source_records = []
        try:
            for month in pd.period_range(pd.Timestamp(days[0]), pd.Timestamp(days[-1]), freq="M"):
                path = store.path("bar1m", str(month))
                if path is None:
                    continue
                source_root = store.training_run if store.training_run in path.parents else store.holdout_run
                relative = path.relative_to(source_root / "data")
                expected = read_json(source_root / "records" / relative.with_suffix(".json"))
                digest = sha256(path)
                if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
                    raise ValueError("盘前预处理来源发生变化: " + str(path))
                source_records.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest})
                month_days = days[(days >= month.start_time.value) & (days < (month + 1).start_time.value)]
                predicate = None
                for stamp in month_days:
                    lo = pd.Timestamp(int(stamp)) + pd.Timedelta(minutes=int(MINUTES[0]))
                    hi = pd.Timestamp(int(stamp)) + pd.Timedelta(minutes=int(MINUTES[-1]) + 1)
                    term = (ds.field("date") >= lo) & (ds.field("date") < hi)
                    predicate = term if predicate is None else predicate | term
                scanner = ds.dataset(path, format="parquet").scanner(
                    columns=KEYS + VALUE_COLUMNS, filter=predicate, batch_size=16384)
                for batch in scanner.to_batches():
                    frame = batch.to_pandas()
                    times = frame.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                    day = np.searchsorted(days, times // DAY_NS * DAY_NS)
                    stock = stocks.get_indexer(frame.instrument)
                    minute = times // MINUTE_NS % 1440
                    slot = np.searchsorted(MINUTES, minute)
                    valid = (day < len(days)) & (stock >= 0) & (slot < BARS) & (times % MINUTE_NS == 0)
                    valid &= days[day.clip(0, len(days) - 1)] == times // DAY_NS * DAY_NS
                    valid &= MINUTES[slot.clip(0, BARS - 1)] == minute
                    take = np.flatnonzero(valid)
                    block = mapping[day[take], stock[take]]
                    take, block = take[block > 0], block[block > 0]
                    flat = block.astype(np.int64) * BARS + slot[take]
                    if np.unique(flat).size != len(flat) or occupied.reshape(-1)[flat].any():
                        raise ValueError("盘前缓存遇到重复股票 Bar")
                    occupied.reshape(-1)[flat] = True
                    values[block, slot[take]] = frame.iloc[take][VALUE_COLUMNS].to_numpy(dtype=np.float32)
            values.flush()
        finally:
            values._mmap.close()
            del values
        files = {}
        for name in ("days.npy", "map.npy", "values.npy"):
            path = staging / name
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path), "shape": list(array.shape),
                           "dtype": array.dtype.str, "array_bytes": array.nbytes}
            array._mmap.close()
        manifest = {"format": FORMAT, "status": "complete", "cache_key": key, "identity": identity,
                    "stocks": stocks.tolist(), "blocks": count, "rows": len(keys),
                    "observed_bars": int(occupied.sum()), "files": files, "sources": source_records,
                    "storage_bytes": sum(v["bytes"] for v in files.values()), "labels_read": False}
        write_json(staging / "manifest.json", manifest)
        try:
            os.rename(staging, directory)
        except FileExistsError:
            saved = read_json(directory / "manifest.json")
            if saved.get("cache_key") != key or saved.get("status") != "complete":
                raise ValueError("并发盘前缓存来源不一致")


class PreopenContext:
    def __init__(self, store, split, report=None):
        started = time.perf_counter()
        self.config = store.config
        self.horizon = store.config["preopen_context"]["previous_days"]
        key, identity = input_identity(store, split)
        self.directory = store.training_run.parent / "preopen-context-cache" / key
        if not self.directory.exists():
            build_cache(store, split, self.directory, key, identity)
        self.manifest = read_json(self.directory / "manifest.json")
        if self.manifest.get("status") != "complete" or self.manifest.get("cache_key") != key or self.manifest.get("identity") != identity:
            raise ValueError("盘前缓存来源或状态不符")
        self.arrays = {}
        try:
            for name in ("days.npy", "map.npy", "values.npy"):
                path = self.directory / name
                record = self.manifest["files"][name]
                if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                    raise ValueError("盘前缓存文件已改变")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                self.arrays[name] = array
                if list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]:
                    raise ValueError("盘前缓存数组类型或形状不符")
        except BaseException:
            self.close()
            raise
        self.stocks = pd.Index(self.manifest["stocks"])
        if report:
            report("preopen_context_cache_ready", split=split, cache=str(self.directory.resolve()),
                   storage_bytes=self.manifest["storage_bytes"], seconds=time.perf_counter() - started)

    def batch(self, dates, instruments):
        days = self.arrays["days.npy"]
        positions = history_positions(days.astype("datetime64[ns]"), dates, self.horizon)
        stocks = self.stocks.get_indexer(instruments)
        if (stocks < 0).any():
            raise ValueError("盘前缓存缺少目标股票")
        block = self.arrays["map.npy"][positions.clip(0), stocks[:, None]]
        block = np.where(positions >= 0, block, 0)
        raw = np.asarray(self.arrays["values.npy"][block], dtype=np.float32)
        return prepare_values(raw, self.config["data"]["input_precision"])

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()


class PreopenBlock:
    """保留短期块所有索引及目标接口，额外暴露独立的历史输入。"""
    def __init__(self, base, reader):
        self.base, self.preopen_reader = base, reader

    def __getattr__(self, name):
        return getattr(self.base, name)

    def preopen_batch(self, indices):
        if hasattr(self.base, "raw_batch"):
            dates = self.base.arrays["days"][self.base.day[indices]].astype("datetime64[ns]")
            stocks = np.asarray(self.base.manifest["stocks"], dtype=object)[self.base.stock[indices]]
        else:
            selected = self.base.keys.iloc[indices]
            dates, stocks = selected.date.to_numpy(), selected.instrument.to_numpy()
        return self.preopen_reader.batch(dates, stocks)

    def batch(self, indices):
        windows, clock = self.base.batch(indices)
        windows[PREOPEN_KEY] = self.preopen_batch(indices)
        return windows, clock


def append_preopen_batches(iterator, block, order, start, batch_size, device, representatives=None):
    """后台准备下一批盘前输入，与现有短期预取及 GPU 更新重叠。"""
    def rows(offset):
        chosen = order[offset:offset + batch_size]
        return chosen if representatives is None else representatives[chosen]
    if start >= len(order):
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(block.preopen_batch, rows(start))
        for offset, indices, windows, clock, targets in iterator:
            context = pending.result()
            following = offset + len(indices)
            pending = pool.submit(block.preopen_batch, rows(following)) if following < len(order) else None
            value = torch.from_numpy(context)
            if device.type == "cuda":
                value = value.pin_memory()
            # 同步完成这次搬运后主机数组即可释放；下一批仍由后台线程准备。
            windows[PREOPEN_KEY] = value.to(device)
            yield offset, indices, windows, clock, targets
