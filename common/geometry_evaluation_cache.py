"""全局训练的验证／测试缓存；保持原输入、分块和 batch 边界。"""

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time

import torch
import numpy as np
import pandas as pd
import psutil

from artifacts import read_json, write_json, sha256, log_event
from baseline_evaluation import evaluate
import geometry_resident as resident
import price_geometry

FORMAT = "jycapital-evaluation-price-geometry-cache-v1"
KEYS = ["date", "instrument"]


class EvaluationPlan(resident.TrainingPlan):
    """完整验证键及其历史并集；历史可以跨训练／验证／测试边界。"""

    def __init__(self, store, split):
        interval = store.protocol["splits"][split]
        self.keys = store.universe.loc[
            (store.universe.date >= interval["start"]) & (store.universe.date < interval["stop"]), KEYS
        ].sort_values(KEYS).reset_index(drop=True)
        if self.keys.empty or self.keys.duplicated(KEYS).any() or self.keys.isna().any().any():
            raise ValueError("评估股票池为空、重复或含空键")
        times = self.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        days = times // resident.DAY_NS * resident.DAY_NS
        calendar = np.asarray(store.days).astype("datetime64[ns]").astype(np.int64)
        start = max(0, int(np.searchsorted(calendar, days.min())) - max(resident.PREVIOUS_DAYS.values()))
        stop = int(np.searchsorted(calendar, days.max())) + 1
        self.days = calendar[start:stop]
        self.stocks = pd.Index(sorted(self.keys.instrument.unique()))
        self.stock = self.stocks.get_indexer(self.keys.instrument).astype(np.int32)
        self.day = np.searchsorted(self.days, days).astype(np.int32)
        minutes = times // resident.MINUTE_NS % 1440
        clock = np.searchsorted(resident.CLOCK_MINUTES, minutes)
        if np.any(clock >= 8) or (times % resident.MINUTE_NS).any() or not np.array_equal(
                resident.CLOCK_MINUTES[clock], minutes):
            raise ValueError("评估键包含非法预测时刻")
        self.clock = clock.astype(np.uint8)
        present = np.zeros((len(self.days), len(self.stocks)), bool)
        present[self.day, self.stock] = True
        self.maps, self.counts = {}, {}
        for frequency in resident.FREQUENCIES:
            needed = present.copy()
            for offset in range(1, resident.PREVIOUS_DAYS[frequency] + 1):
                needed[:-offset] |= present[offset:]
            stocks, days = np.nonzero(needed.T)
            mapping = np.zeros(present.shape, np.int32)
            mapping[days, stocks] = np.arange(1, len(days) + 1, dtype=np.int32)
            self.maps[frequency], self.counts[frequency] = mapping, len(days) + 1


def cache_key(store, split, data_manifest):
    identity = {"format": FORMAT, "split": split, "protocol": store.protocol,
                "data": store.config["data"], "sources": data_manifest,
                "builder": sha256(__file__), "packing": sha256(resident.__file__),
                "geometry": sha256(price_geometry.__file__),
                "reference_loader": inspect.getsource(type(store).load_range)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def build_evaluation_cache(store, split, directory, key, report=None):
    """复用已有打包器，原始标签单独保存；不筛除任何预测键。"""
    directory = Path(directory)
    if directory.exists():
        saved = read_json(directory / "evaluation.json")
        if saved.get("format") != FORMAT or saved.get("cache_key") != key or saved.get("split") != split:
            raise ValueError("评估缓存来源或划分不匹配")
        return saved
    plan = EvaluationPlan(store, split)
    proxy = copy.copy(store)
    proxy.protocol = copy.deepcopy(store.protocol)
    proxy.protocol["splits"]["train"] = dict(store.protocol["splits"][split])
    # 文件路径仍由原协议解析；额外月份只提供因果历史，不产生评估样本。
    proxy.path = store.path
    proxy.months = lambda _: [str(m) for m in pd.period_range(
        pd.Timestamp(plan.days[0]), pd.Timestamp(plan.days[-1]), freq="M")]
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=key[:12] + "-", dir=directory.parent) as temporary:
        staging = Path(temporary)

        def save_labels(keys, protocol):
            keys.to_parquet(staging / "keys.parquet", index=False)
            # 批次加载器的占位目标；评估使用 keys 中原始 FP64 标签，包括 NaN。
            return np.zeros(len(keys), np.float64), {}

        packed = resident.build_cache(proxy, staging / "data", key, save_labels, plan=plan, report=report)
        saved = {"format": FORMAT, "cache_key": key, "split": split,
                 "rows": len(plan.keys), "resident_array_bytes": packed["resident_array_bytes"],
                 "keys_sha256": sha256(staging / "keys.parquet"),
                 "packed_manifest_sha256": sha256(staging / "data/manifest.json")}
        write_json(staging / "evaluation.json", saved)
        # 只发布完整缓存；构建中断不留下可被误用的正式目录。
        try:
            os.rename(staging, directory)
        except FileExistsError:
            other = read_json(directory / "evaluation.json")
            if other != saved:
                raise ValueError("并发构建的评估缓存内容不同")
    return saved


class EvaluationData:
    """预算充足时常驻全部数值，否则按原评估块读取紧凑数组。"""

    def __init__(self, directory, key, mode, reserve_gib, batch_size):
        if mode not in ("auto", "resident", "stream"):
            raise ValueError("未知评估缓存加载方式")
        self.directory, self.reserve_gib = Path(directory), reserve_gib
        saved = read_json(self.directory / "evaluation.json")
        if saved.get("format") != FORMAT or saved.get("cache_key") != key:
            raise ValueError("评估缓存标识不符")
        packed = self.directory / "data"
        if sha256(packed / "manifest.json") != saved["packed_manifest_sha256"]:
            raise ValueError("评估缓存清单已改变")
        manifest = read_json(packed / "manifest.json")
        if manifest.get("status") != "complete" or manifest.get("cache_key") != key:
            raise ValueError("评估缓存未完成")
        if sha256(self.directory / "keys.parquet") != saved["keys_sha256"]:
            raise ValueError("评估缓存标签／键已改变")
        self.keys = pd.read_parquet(self.directory / "keys.parquet")
        if len(self.keys) != saved["rows"] or self.keys.duplicated(KEYS).any():
            raise ValueError("评估缓存键不完整")
        self.arrays, self.mapped = {}, {}
        values_bytes = sum(r["array_bytes"] for n, r in manifest["files"].items() if n.endswith("-values.npy")
                           and "exception" not in n)
        metadata_bytes = manifest["resident_array_bytes"] - values_bytes
        self.buffer_bytes = 2 * batch_size * sum((resident.PREVIOUS_DAYS[f] + 1) * resident.DAILY_BARS[f]
                                                   for f in resident.FREQUENCIES) * 72 * 2
        resident.require_memory(metadata_bytes + self.buffer_bytes, reserve_gib)
        for name, record in manifest["files"].items():
            path = packed / name
            if path.parent != packed or path.suffix != ".npy":
                raise ValueError("非法评估缓存文件路径")
            if name.endswith("-values.npy") and "exception" not in name:
                continue
            self.arrays[name[:-4]] = resident._load_resident_array(path, record)
        enough = psutil.virtual_memory().available >= values_bytes + self.buffer_bytes + reserve_gib * 2**30
        self.mode = "resident" if mode == "resident" or mode == "auto" and enough else "stream"
        if self.mode == "resident":
            resident.require_memory(values_bytes + self.buffer_bytes, reserve_gib)
        for frequency in resident.FREQUENCIES:
            name = frequency + "-values.npy"
            path, record = packed / name, manifest["files"][name]
            if self.mode == "resident":
                self.arrays[name[:-4]] = resident._load_resident_array(path, record)
            else:
                if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                    raise ValueError("评估数值缓存校验失败: " + name)
                mapped = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(mapped.shape) != record["shape"] or mapped.dtype.str != record["dtype"]:
                    raise ValueError("评估缓存数组形状或类型不同")
                self.mapped[name[:-4]] = mapped
        self.base = resident.ResidentBlock.__new__(resident.ResidentBlock)
        self.base.manifest, self.base.arrays = manifest, self.arrays
        self.base.keys, self.base.targets = self.keys, self.arrays["targets"]
        self.base.day, self.base.stock = self.arrays["sample_day"], self.arrays["sample_stock"]
        self.base.clock = self.arrays["sample_clock"]
        self.base.day_plans = {f: np.arange(len(self.arrays["days"]))[:, None] + np.arange(
            -resident.PREVIOUS_DAYS[f], 1) for f in resident.FREQUENCIES}

    def load_segment(self, order):
        if self.mode == "resident":
            return self.base
        selections = {}
        for f in resident.FREQUENCIES:
            days = self.base.day_plans[f][self.base.day[order]]
            blocks = self.arrays[f + "-map"][days.clip(0), self.base.stock[order, None]]
            selections[f] = np.unique(np.r_[0, np.where(days >= 0, blocks, 0).reshape(-1)])
        size = sum(len(ids) * resident.DAILY_BARS[f] * 72 * 2 for f, ids in selections.items())
        resident.require_memory(size + self.buffer_bytes, self.reserve_gib)
        block = copy.copy(self.base)
        block.arrays = dict(self.arrays)
        for f, ids in selections.items():
            values = np.empty((len(ids), resident.DAILY_BARS[f], 72), np.float16)
            np.take(self.mapped[f + "-values"], ids, axis=0, out=values, mode="clip")
            values.flags.writeable = False
            remap = np.zeros(len(self.mapped[f + "-values"]), np.int32)
            remap[ids] = np.arange(len(ids), dtype=np.int32)
            block.arrays[f + "-map"] = remap[self.arrays[f + "-map"]]
            block.arrays[f + "-values"] = values
            old = self.arrays[f + "-exception-pointers"]
            offsets, repairs, pointers = [], [], [0]
            for index in ids:
                lo, hi = old[index:index + 2]
                offsets.append(self.arrays[f + "-exception-offsets"][lo:hi])
                repairs.append(self.arrays[f + "-exception-values"][lo:hi])
                pointers.append(pointers[-1] + int(hi - lo))
            block.arrays[f + "-exception-pointers"] = np.asarray(pointers, np.int64)
            block.arrays[f + "-exception-offsets"] = np.concatenate(offsets)
            block.arrays[f + "-exception-values"] = np.concatenate(repairs)
        return block

    def close(self):
        for value in self.mapped.values():
            value._mmap.close()
        self.mapped.clear()


def predict_cached_split(store, split, model, device, config, run):
    """保持原 5 日块的尾批与完整评估键；模型计算仍为 FP32。"""
    started = time.perf_counter()
    mode = getattr(store, "evaluation_cache_mode", "auto")
    active = getattr(store, "evaluation_cache", None)
    if active is None or active[0] != split:
        if active is not None:
            active[1].close()
            store.evaluation_cache = None
            del active
        original = store.original
        key = cache_key(original, split, read_json(Path(run) / "data-manifest.json"))
        directory = original.training_run.parent / "evaluation-cache" / key
        report = lambda event, **fields: log_event(run, "evaluation_" + event, split=split, **fields)
        build_evaluation_cache(original, split, directory, key, report)
        data = EvaluationData(directory, key, mode, config["training"]["resident_cache"]["memory_reserve_gib"],
                              config["inference"]["batch_size"])
        store.evaluation_cache = (split, data)
        log_event(run, "evaluation_cache_ready", split=split, mode=data.mode, cache=str(directory.resolve()),
                  rows=len(data.keys), seconds=time.perf_counter() - started)
    data = store.evaluation_cache[1]
    # 保留原块顺序和每块尾批，避免改变 GPU batch 形状影响数值可比性。
    times = data.keys.date.to_numpy(dtype="datetime64[ns]")
    segments = [(name, np.arange(np.searchsorted(times, np.datetime64(lo)),
                                np.searchsorted(times, np.datetime64(hi))))
                for name, (lo, hi) in store.segments[split].items()]
    predictions = np.empty(len(data.keys), np.float64)
    model.eval()
    prediction_started = time.perf_counter()
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(data.load_segment, segments[0][1])
        for number, (name, order) in enumerate(segments):
            block = pending.result()
            if number + 1 < len(segments):
                pending = pool.submit(data.load_segment, segments[number + 1][1])
            iterator = resident.ResidentBatches(block, order, block.targets, 0,
                                               config["inference"]["batch_size"], device)
            for _, indices, windows, clock, _ in iterator:
                # 与旧 CPU 路径的连续 NCL 布局相同，不改变模型精度或推理后端。
                windows = {f: x.contiguous() for f, x in windows.items()}
                predictions[indices] = (model(windows, clock) / config["inference"]["factor_scale"]).cpu().numpy()
            del iterator, block
            log_event(run, "predicted_month", split=split, month=name, rows=len(order))
    frame = data.keys[KEYS].copy()
    frame["factor"] = predictions
    metrics_started = time.perf_counter()
    result = evaluate(frame, data.keys, store.protocol)
    log_event(run, "cached_evaluation_finished", split=split, mode=data.mode,
              prediction_seconds=metrics_started - prediction_started,
              metric_seconds=time.perf_counter() - metrics_started, seconds=time.perf_counter() - started)
    return frame, data.keys, result
