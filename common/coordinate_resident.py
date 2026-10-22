"""共同数量坐标旁路与紧凑常驻输入；父缓存保持只读。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch
import numpy as np
import pandas as pd
import psutil

from artifacts import read_json, write_json, sha256
import geometry_resident as base
import coordinate_inputs as inputs
from coordinate_inputs import (FREQUENCIES, DAILY_BARS, PREVIOUS_DAYS, PRECISION,
    raw_batches, transform_side, prepare_coordinate_windows, base_data_manifest, parameter_digest)

FORMAT = "jycapital-resident-coordinate-view-v1"
MINUTE_NS, DAY_NS, CLOCK_MINUTES = base.MINUTE_NS, base.DAY_NS, base.CLOCK_MINUTES
require_memory = base.require_memory


def validate_resident_config(config):
    inputs.validate_coordinate_config(config)
    mode = config["training"].get("batching")
    if mode == "monthly_shuffled_samples":
        if config["data"]["input_precision"] != "float32":
            raise ValueError("月内参考模式须使用 FP32 输入")
        return
    spec = config["training"]["resident_cache"]
    if mode != "global_shuffled_samples" or config["data"]["input_precision"] != PRECISION:
        raise ValueError("全局坐标训练要求指定混合精度")
    if spec.get("format") != FORMAT or spec.get("prefetch_batches") != 2:
        raise ValueError("坐标缓存格式或预取配置不匹配")
    for key in ("build_batch_rows", "evaluation_days"):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError(key + " 必须为正整数")
    if spec.get("memory_reserve_gib") != 2.0:
        raise ValueError("本系列按用户要求预留 2 GiB")


def base_key(store, manifest, transform):
    return base.cache_key(store.base, base_data_manifest(manifest), transform)


def side_key(parent_key, parameters):
    value = {"format": FORMAT, "base_key": parent_key, "parameters": parameters,
             "contract": inputs.CONTRACT, "builder_sha256": sha256(__file__),
             "input_code_sha256": sha256(inputs.__file__)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def cache_key(store, manifest, transform):
    return side_key(base_key(store, manifest, transform), store.parameters)


def _safe_array_path(directory, name):
    path = Path(directory) / name
    if Path(name).name != name or path.suffix != ".npy":
        raise ValueError("非法缓存数组路径")
    return path


def _mmap_verified(path, record):
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise ValueError("缓存文件校验失败: " + path.name)
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != record["shape"] or value.dtype.str != record["dtype"]:
        value._mmap.close()
        raise ValueError("缓存数组布局不一致")
    return value


def _compact_parent_values(path, record):
    """边校验边保留前52列，不同时分配父72列完整数组。"""
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        if list(shape) != record["shape"] or len(shape) != 3 or shape[-1] != 72 or fortran or dtype != np.dtype("<f2"):
            raise ValueError("父缓存数值布局不匹配")
        if math.prod(shape) * dtype.itemsize != record["array_bytes"]:
            raise ValueError("父缓存数值大小不匹配")
        header_end = stream.tell()
        stream.seek(0)
        digest = hashlib.sha256(stream.read(header_end))
        output = np.empty((*shape[:-1], 52), np.float16)
        flat = output.reshape(-1, 52)
        block_rows = min(len(flat), 131072)
        buffer = np.empty((block_rows, 72), np.float16)
        for start in range(0, len(flat), block_rows):
            count = min(block_rows, len(flat) - start)
            view = memoryview(buffer[:count]).cast("B")
            position = 0
            while position < len(view):
                copied = stream.readinto(view[position:])
                if not copied: raise ValueError("父缓存数值文件被截断")
                position += copied
            digest.update(view)
            flat[start:start + count] = buffer[:count, :52]
        if stream.read(1) or digest.hexdigest() != record["sha256"]:
            raise ValueError("父缓存数值哈希不一致")
    output.flags.writeable = False
    return output


def _record_array(path):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    result = {"bytes": path.stat().st_size, "sha256": sha256(path), "shape": list(value.shape),
              "dtype": value.dtype.str, "array_bytes": int(value.nbytes)}
    value._mmap.close()
    return result


def _remove_own_staging(staging, destination):
    actual, parent = staging.resolve(), destination.parent.resolve()
    if actual.parent != parent or not actual.name.startswith(destination.name + ".building-") or staging.is_symlink():
        raise ValueError("临时缓存清理范围不安全")
    shutil.rmtree(actual)


def ensure_side_cache(store, parent_directory, parent_key, manifest, *, split, report=None):
    parent_directory = Path(parent_directory)
    parent = read_json(parent_directory / "manifest.json")
    if parent.get("format") != base.FORMAT or parent.get("status") != "complete" or parent.get("cache_key") != parent_key:
        raise ValueError("基础输入缓存身份不一致")
    key = side_key(parent_key, store.parameters)
    directory = store.training_run.parent / "coordinate-cache" / key
    base_fingerprints = {name: item["sha256"] for name, item in parent["files"].items()}
    if directory.exists():
        saved = read_json(directory / "manifest.json")
        if saved.get("status") != "complete" or saved.get("format") != FORMAT or saved.get("cache_key") != key or saved.get("base_files") != base_fingerprints:
            raise ValueError("坐标旁路未完成或基础数组身份改变")
        if any(saved.get(name) != parent.get(name) for name in ("stocks", "catalog", "frequencies")):
            raise ValueError("坐标旁路的基础结构元数据改变")
        return directory, key
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=directory.name + ".building-", dir=directory.parent))
    opened = []
    try:
        days = base._load_resident_array(parent_directory / "days.npy", parent["files"]["days.npy"])
        days = days.astype("datetime64[ns]").astype(np.int64)
        stocks = pd.Index(parent["stocks"])
        available_sources = {item["path"] for item in base_data_manifest(manifest)["files"]}
        months = [str(month) for month in pd.DatetimeIndex(days).to_period("M").unique()]
        for frequency in FREQUENCIES:
            values_record = parent["files"][frequency + "-values.npy"]
            blocks, bars, width = values_record["shape"]
            if width != 72 or bars != DAILY_BARS[frequency]:
                raise ValueError("父缓存频率布局不一致")
            mapping = base._load_resident_array(parent_directory / (frequency + "-map.npy"), parent["files"][frequency + "-map.npy"])
            quantities = np.lib.format.open_memmap(staging / (frequency + "-quantity-values.npy"), mode="w+", dtype=np.float16, shape=(blocks, bars, 20))
            opened.append(quantities)
            anchors = np.lib.format.open_memmap(staging / (frequency + "-anchor-values.npy"), mode="w+", dtype=np.float32, shape=(blocks, bars))
            opened.append(anchors)
            quantities[:] = np.nan
            anchors[:] = np.nan
            seen = np.zeros(blocks * bars, bool)
            overflow_positions, overflow_values = [], []
            step = int(frequency[3:-1])
            minutes = np.r_[np.arange(570 + step, 691, step), np.arange(780 + step, 901, step)]
            eligible_months = [month for month in months if f"raw/cpt_jyc_2026_stock_{frequency}/{month}.parquet" in available_sources]
            for frame in raw_batches(store.data_root, frequency, eligible_months, pd.Timestamp(days[0]),
                                     pd.Timestamp(days[-1]) + pd.Timedelta(days=1), stocks=stocks,
                                     batch_rows=store.config["training"]["resident_cache"]["build_batch_rows"]):
                times = frame.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                day = times // DAY_NS * DAY_NS
                di = np.searchsorted(days, day)
                si = stocks.get_indexer(frame.instrument.astype(str))
                minute = times // MINUTE_NS % 1440
                slot = np.searchsorted(minutes, minute)
                valid = (di < len(days)) & (si >= 0) & (slot < bars) & (times % MINUTE_NS == 0)
                valid &= days[di.clip(0, len(days)-1)] == day
                valid &= minutes[slot.clip(0, bars-1)] == minute
                positions = np.flatnonzero(valid)
                block_ids = mapping[di[positions], si[positions]]
                positions, block_ids = positions[block_ids > 0], block_ids[block_ids > 0]
                local_slots = slot[positions]
                rows = block_ids.astype(np.int64) * bars + local_slots
                if len(np.unique(rows)) != len(rows) or seen[rows].any():
                    raise ValueError("原始坐标旁路存在重复网格位置")
                side, anchor = transform_side(frame.iloc[positions], store.parameters[frequency])
                with np.errstate(over="ignore", invalid="ignore"):
                    half = side.astype(np.float16)
                over = np.isfinite(side) & ~np.isfinite(half)
                rr, cc = np.nonzero(over)
                if len(rr):
                    overflow_positions.append(rows[rr] * 20 + cc)
                    overflow_values.append(side[rr, cc])
                quantities[block_ids, local_slots] = half
                anchors[block_ids, local_slots] = anchor
                seen[rows] = True
            flat = np.concatenate(overflow_positions) if overflow_positions else np.empty(0, np.int64)
            exceptional = np.concatenate(overflow_values) if overflow_values else np.empty(0, np.float32)
            order = np.argsort(flat, kind="stable")
            flat, exceptional = flat[order], exceptional[order]
            ids = flat // (bars * 20)
            pointers = np.r_[0, np.bincount(ids, minlength=blocks).cumsum()].astype(np.int64)
            np.save(staging / (frequency + "-quantity-exception-pointers.npy"), pointers, allow_pickle=False)
            np.save(staging / (frequency + "-quantity-exception-offsets.npy"), (flat % (bars * 20)).astype(np.int32), allow_pickle=False)
            np.save(staging / (frequency + "-quantity-exception-values.npy"), exceptional, allow_pickle=False)
            quantities.flush(); anchors.flush()
            quantities._mmap.close(); anchors._mmap.close()
            opened.clear()
            if report:
                report("coordinate_frequency_built", cache_split=split, frequency=frequency,
                       observed_bars=int(seen.sum()), overflow_values=len(flat))
            del quantities, anchors, mapping, seen
        files = {path.name: _record_array(path) for path in sorted(staging.glob("*.npy"))}
        parent_feature_bytes = sum(parent["files"][f + "-values.npy"]["array_bytes"] for f in FREQUENCIES)
        retained_bytes = parent_feature_bytes // 72 * 52
        resident_bytes = parent["resident_array_bytes"] - parent_feature_bytes + retained_bytes + sum(v["array_bytes"] for v in files.values())
        saved = {"format": FORMAT, "cache_key": key, "status": "complete", "split": split,
            "base_cache_key": parent_key, "base_directory": str(parent_directory.resolve()),
            "base_manifest_sha256_at_build": sha256(parent_directory / "manifest.json"),
            "base_files": base_fingerprints, "parameters": store.parameters,
            "parameters_sha256": parameter_digest(store.parameters), "contract": inputs.CONTRACT,
            "files": files, "catalog": parent["catalog"], "input_precision": PRECISION,
            "stocks": parent["stocks"], "frequencies": parent["frequencies"],
            "resident_array_bytes": resident_bytes, "base_retained_channels": 52,
            "side_array_bytes": sum(v["array_bytes"] for v in files.values()),
            "label_values_read_by_side_builder": False, "parent_cache_modified": False}
        write_json(staging / "manifest.json", saved)
        parent_scope = directory.parent.resolve()
        if staging.resolve().parent != parent_scope or directory.resolve().parent != parent_scope:
            raise ValueError("缓存发布路径超出预期目录")
        os.replace(staging, directory)
        return directory, key
    except BaseException:
        for value in opened:
            if getattr(value, "_mmap", None) is not None:
                value._mmap.close()
        if staging.exists(): _remove_own_staging(staging, directory)
        raise


class ResidentBlock:
    """训练紧凑驻留52父通道；流式评估可以只映射原父数组。"""
    def __init__(self, directory, key, reserve_gib=2.0, report=None, *, resident=True):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        m = self.manifest
        if m.get("status") != "complete" or m.get("format") != FORMAT or m.get("cache_key") != key:
            raise ValueError("坐标缓存身份不一致")
        if m.get("contract") != inputs.CONTRACT or parameter_digest(m["parameters"]) != m.get("parameters_sha256") or side_key(m["base_cache_key"], m["parameters"]) != key:
            raise ValueError("坐标缓存参数或实现身份改变")
        parent_dir = Path(m["base_directory"])
        parent = read_json(parent_dir / "manifest.json")
        if parent["cache_key"] != m["base_cache_key"] or {n:r["sha256"] for n,r in parent["files"].items()} != m["base_files"]:
            raise ValueError("坐标缓存的基础数组改变")
        if any(m.get(name) != parent.get(name) for name in ("stocks", "catalog", "frequencies")):
            raise ValueError("坐标缓存的基础结构元数据改变")
        self.parameters = m["parameters"]
        self.compact = bool(resident)
        self._mapped = []
        arrays = {}
        numeric_names = {f + "-values.npy" for f in FREQUENCIES}
        side_numeric = {f + suffix for f in FREQUENCIES for suffix in ("-quantity-values.npy", "-anchor-values.npy")}
        metadata = sum(v["array_bytes"] for n,v in parent["files"].items() if n not in numeric_names)
        metadata += sum(v["array_bytes"] for n,v in m["files"].items() if n not in side_numeric)
        require_memory((m["resident_array_bytes"] if resident else metadata) + m["catalog"]["rows"] * 8, reserve_gib)
        try:
            for name, record in parent["files"].items():
                path = _safe_array_path(parent_dir, name)
                if name in numeric_names:
                    value = _compact_parent_values(path, record) if resident else _mmap_verified(path, record)
                    if not resident: self._mapped.append(value)
                else:
                    value = base._load_resident_array(path, record)
                arrays[name[:-4]] = value
            # 父异常坐标的局部行跨度仍是72，保留前52列即可直接修复拼接后的编码。
            for f in FREQUENCIES:
                offsets = arrays[f + "-exception-offsets"]
                keep = np.flatnonzero(offsets % 72 < 52)
                arrays[f + "-exception-pointers"] = np.searchsorted(keep, arrays[f + "-exception-pointers"]).astype(np.int64)
                arrays[f + "-exception-offsets"] = offsets[keep]
                arrays[f + "-exception-values"] = arrays[f + "-exception-values"][keep]
            for name, record in m["files"].items():
                path = _safe_array_path(self.directory, name)
                if not resident and name in side_numeric:
                    value = _mmap_verified(path, record)
                    self._mapped.append(value)
                else:
                    value = base._load_resident_array(path, record)
                arrays[name[:-4]] = value
            self.arrays = arrays
            self.targets, self.clock = arrays["targets"], arrays["sample_clock"]
            self.day, self.stock = arrays["sample_day"], arrays["sample_stock"]
            self.day_plans = {f: np.arange(len(arrays["days"]))[:, None] + np.arange(-PREVIOUS_DAYS[f], 1)
                              for f in FREQUENCIES}
            if report: report("coordinate_arrays_loaded", resident=resident, array_bytes=m["resident_array_bytes"])
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in self._mapped:
            value._mmap.close()
        self._mapped.clear()

    def training_targets(self):
        return self.targets, {stamp: value["reason"] for stamp,value in self.manifest["catalog"]["skipped"].items()}

    def raw_batch(self, indices, destinations=None):
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self.targets)).any():
            raise ValueError("非法样本索引")
        count, raw, repairs = len(indices), {}, {}
        for f in FREQUENCIES:
            days = self.day_plans[f][self.day[indices]]
            block = self.arrays[f + "-map"][days.clip(0), self.stock[indices, None]]
            block = np.where(days >= 0, block, 0)
            length = days.shape[1] * DAILY_BARS[f]
            if destinations is None:
                codes, anchors = np.empty((count, length, 72), np.float16), np.empty((count, length), np.float32)
            else:
                codes, anchors = (value[:count] for value in destinations[f])
            part = np.take(self.arrays[f + "-values"], block.reshape(-1), axis=0)
            codes[..., :52] = part[..., :52].reshape(count, length, 52)
            codes[..., 52:] = np.take(self.arrays[f + "-quantity-values"], block.reshape(-1), axis=0).reshape(count, length, 20)
            anchors[:] = np.take(self.arrays[f + "-anchor-values"], block.reshape(-1), axis=0).reshape(count, length)
            positions, values = [], []
            for prefix, quantity in ((f, False), (f + "-quantity", True)):
                pointers = self.arrays[prefix + "-exception-pointers"]
                starts, ends = pointers[block], pointers[block + 1]
                for row, slot in zip(*np.nonzero(ends > starts)):
                    start, stop = starts[row, slot], ends[row, slot]
                    local = self.arrays[prefix + "-exception-offsets"][start:stop].astype(np.int64)
                    if quantity: local = local // 20 * 72 + 52 + local % 20
                    positions.append(row * length * 72 + slot * DAILY_BARS[f] * 72 + local)
                    values.append(self.arrays[prefix + "-exception-values"][start:stop])
            repairs[f] = (np.concatenate(positions) if positions else np.empty(0, np.int64),
                          np.concatenate(values) if values else np.empty(0, np.float32))
            raw[f] = codes, anchors
        return raw, repairs, np.eye(8, dtype=np.float32)[self.clock[indices]]

    def batch(self, indices):
        raw, repairs, clock = self.raw_batch(indices)
        raw = {f: tuple(torch.from_numpy(value) for value in pair) for f,pair in raw.items()}
        fixed = {f: tuple(torch.from_numpy(value) for value in pair) for f,pair in repairs.items()}
        windows = prepare_coordinate_windows(raw, fixed, torch.from_numpy(clock), self.parameters)
        return {f: value.numpy() for f,value in windows.items()}, clock


class ResidentBatches:
    def __init__(self, block, order, targets, start, batch_size, device):
        self.block, self.order, self.targets = block, order, targets
        self.start, self.batch_size, self.device = start, batch_size, torch.device(device)

    def __iter__(self):
        cuda = self.device.type == "cuda"
        stream = torch.cuda.Stream(device=self.device) if cuda else None
        parameters = self.block.parameters if hasattr(self.block, "parameters") else self.block.base.parameters
        slots = []
        lengths = {f: (PREVIOUS_DAYS[f] + 1) * DAILY_BARS[f] for f in FREQUENCIES}
        for _ in range(2):
            tensors = {f: (torch.empty(self.batch_size, length, 72, dtype=torch.float16, pin_memory=cuda),
                           torch.empty(self.batch_size, length, dtype=torch.float32, pin_memory=cuda)) for f,length in lengths.items()}
            slots.append({"tensors": tensors, "numpy": {f: tuple(v.numpy() for v in pair) for f,pair in tensors.items()}, "event": None})
        offsets = iter(range(self.start, len(self.order), self.batch_size))

        def fill(offset, slot):
            if slot["event"] is not None: slot["event"].synchronize()
            selected = self.order[offset:offset + self.batch_size]
            raw, repairs, clock = self.block.raw_batch(selected, slot["numpy"])
            return offset, selected, slot, raw, repairs, clock

        def transfer(packed):
            offset, selected, slot, raw, repairs, clock = packed
            def copy_and_prepare():
                values = {f: tuple(v[:len(selected)].to(self.device, non_blocking=cuda) for v in slot["tensors"][f]) for f in FREQUENCIES}
                fixed = {f: tuple(torch.from_numpy(v).to(self.device, non_blocking=cuda) for v in pair) for f,pair in repairs.items()}
                clocks = torch.from_numpy(clock).to(self.device, non_blocking=cuda)
                target = torch.from_numpy(self.targets[selected]).to(self.device, non_blocking=cuda)
                return prepare_coordinate_windows(values, fixed, clocks, parameters), clocks, target
            if cuda:
                with torch.cuda.stream(stream):
                    values = copy_and_prepare()
                    ready = torch.cuda.Event(); ready.record(stream)
                slot["event"] = ready
            else:
                values, ready = copy_and_prepare(), None
            return offset, selected, values, ready, packed

        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                first = next(offsets, None)
                if first is None: return
                current = transfer(pool.submit(fill, first, slots[0]).result())
                number, offset = 1, next(offsets, None)
                pending = pool.submit(fill, offset, slots[1]) if offset is not None else None
                while current is not None:
                    upcoming = transfer(pending.result()) if pending is not None else None
                    number += 1
                    offset = next(offsets, None)
                    pending = pool.submit(fill, offset, slots[number % 2]) if offset is not None else None
                    position, selected, (windows, clock, target), ready, host = current
                    if cuda:
                        compute = torch.cuda.current_stream(self.device); compute.wait_event(ready)
                        for value in (*windows.values(), clock, target): value.record_stream(compute)
                    yield position, selected, windows, clock, target
                    current = upcoming
            finally:
                if stream is not None: stream.synchronize()


class ResidentStore(base.ResidentStore):
    pass


def open_resident_store(store, run, transform, directory=None, report=None):
    manifest = read_json(Path(run) / "data-manifest.json")
    parent_key = base_key(store, manifest, transform)
    parent_directory = store.training_run.parent / "resident-cache" / parent_key
    started = time.perf_counter()
    if not parent_directory.exists():
        plan = base.TrainingPlan(store.base)
        base.build_cache(store.base, parent_directory, parent_key, transform, plan=plan, report=report)
        del plan
    selected, key = ensure_side_cache(store, parent_directory, parent_key, manifest, split="train", report=report)
    if directory is not None and Path(directory).resolve() != selected.resolve() and report:
        report("resident_cache_contract_changed", legacy_cache=str(Path(directory).resolve()), selected_cache=str(selected.resolve()), legacy_modified=False)
    spec = store.config["training"]["resident_cache"]
    block = ResidentBlock(selected, key, spec["memory_reserve_gib"], report)
    record = {"format": FORMAT, "cache_key": key, "path": str(selected.resolve()),
        "manifest_sha256": sha256(selected / "manifest.json"), "input_precision": PRECISION,
        "resident_array_bytes": block.manifest["resident_array_bytes"],
        "base_cache_key": parent_key, "base_cache_path": str(parent_directory.resolve()),
        "base_cache_modified": False, "retained_base_channels": 52, "additional_side_array_bytes": block.manifest["side_array_bytes"],
        "load_and_build_seconds": time.perf_counter() - started}
    path = Path(run) / "resident-cache.json"
    if path.exists() and read_json(path)["cache_key"] != key:
        raise ValueError("恢复时坐标缓存身份改变")
    write_json(path, record)
    return ResidentStore(store, block, spec["evaluation_days"])
