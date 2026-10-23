"""父几何、复用价格中心与新增成交参考数组的常驻／流式输入。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import torch
import numpy as np
import pandas as pd

from artifacts import read_json, write_json, sha256
import geometry_resident as base
import coordinate_resident as old_coordinate
import coordinate_inputs as old_inputs
import trade_reference_inputs as inputs
from trade_reference_inputs import (FREQUENCIES, DAILY_BARS, PREVIOUS_DAYS, PRECISION,
    raw_batches, anchor_view, reference_view, prepare_trade_reference_windows, base_data_manifest)

FORMAT = 'jycapital-resident-trade-reference-v1'
MINUTE_NS, DAY_NS, CLOCK_MINUTES = base.MINUTE_NS, base.DAY_NS, base.CLOCK_MINUTES
require_memory = base.require_memory


def validate_resident_config(config):
    inputs.validate_trade_reference_config(config)
    mode = config['training'].get('batching')
    if mode == 'monthly_shuffled_samples':
        if config['data']['input_precision'] != 'float32':
            raise ValueError('月内参考模式须使用 FP32 输入')
        return
    spec = config['training']['resident_cache']
    if mode != 'global_shuffled_samples' or config['data']['input_precision'] != PRECISION:
        raise ValueError('价格历史训练要求指定混合精度')
    if spec.get('format') != FORMAT or spec.get('prefetch_batches') != 2:
        raise ValueError('价格历史缓存格式或预取配置不匹配')
    for key in ('build_batch_rows', 'evaluation_days'):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError(key + ' 必须为正整数')
    if spec.get('memory_reserve_gib') != 2.0:
        raise ValueError('本系列按用户要求预留 2 GiB')


def base_key(store, manifest, transform):
    return base.cache_key(store.base, base_data_manifest(manifest), transform)


def side_key(parent_key):
    value = {'format': FORMAT, 'base_key': parent_key, 'contract': inputs.CONTRACT,
             'builder_sha256': sha256(__file__), 'input_code_sha256': sha256(inputs.__file__), 'parent_input_sha256': sha256(inputs.parent.__file__)}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def cache_key(store, manifest, transform):
    return side_key(base_key(store, manifest, transform))


def reusable_anchor_source(store, parent):
    """只认已冻结且基础数组逐项相同的旧旁路；不读取数量数组。"""
    candidates = store.training_run.parent / 'coordinate-cache'
    fingerprints = {name: item['sha256'] for name, item in parent['files'].items()}
    for path in sorted(candidates.glob('*/manifest.json')):
        if not re.fullmatch('[0-9a-f]{64}', path.parent.name):
            continue
        saved = read_json(path)
        if saved.get('status') != 'complete' or saved.get('base_cache_key') != parent['cache_key']:
            continue
        if saved.get('format') != old_coordinate.FORMAT or saved.get('contract') != old_inputs.CONTRACT:
            continue
        if saved.get('base_files') != fingerprints:
            continue
        if any(saved.get(name) != parent.get(name) for name in ('stocks', 'catalog', 'frequencies')):
            continue
        if old_inputs.parameter_digest(saved['parameters']) != saved.get('parameters_sha256') or old_coordinate.side_key(parent['cache_key'], saved['parameters']) != saved.get('cache_key'):
            raise ValueError('已有价格中心来源身份不一致')
        records = {}
        for f in FREQUENCIES:
            name = f + '-anchor-values.npy'
            record = saved['files'][name]
            if record['shape'] != parent['files'][f + '-values.npy']['shape'][:2] or record['dtype'] != '<f4':
                raise ValueError('已有价格中心布局不一致')
            array = old_coordinate._mmap_verified(path.parent / name, record)
            array._mmap.close()
            records[name] = record
        return path.parent, records, {'kind': 'reused_frozen_coordinate_anchor',
            'source_manifest_sha256': sha256(path), 'source_cache_key': saved['cache_key'],
            'quantity_arrays_read': False}
    return None


def build_anchor_arrays(store, parent_directory, parent, manifest, staging, report, split):
    """独立重建仅写三份 FP32 中心，不要求构建数量旁路。"""
    days = base._load_resident_array(parent_directory / 'days.npy', parent['files']['days.npy'])
    days = days.astype('datetime64[ns]').astype(np.int64)
    stocks = pd.Index(parent['stocks'])
    sources = {item['path'] for item in base_data_manifest(manifest)['files']}
    months = [str(month) for month in pd.DatetimeIndex(days).to_period('M').unique()]
    for f in FREQUENCIES:
        blocks, bars, width = parent['files'][f + '-values.npy']['shape']
        if width != 72 or bars != DAILY_BARS[f]:
            raise ValueError('父缓存频率布局不一致')
        mapping = base._load_resident_array(parent_directory / (f + '-map.npy'), parent['files'][f + '-map.npy'])
        anchors = np.lib.format.open_memmap(staging / (f + '-anchor-values.npy'), mode='w+', dtype=np.float32, shape=(blocks, bars))
        try:
            anchors[:] = np.nan
            seen = np.zeros(blocks * bars, bool)
            step = int(f[3:-1])
            minutes = np.r_[np.arange(570 + step, 691, step), np.arange(780 + step, 901, step)]
            available_months = [m for m in months if f'raw/cpt_jyc_2026_stock_{f}/{m}.parquet' in sources]
            for frame in raw_batches(store.data_root, f, available_months, pd.Timestamp(days[0]),
                                     pd.Timestamp(days[-1]) + pd.Timedelta(days=1), stocks=stocks,
                                     batch_rows=store.config['training']['resident_cache']['build_batch_rows']):
                times = frame.date.to_numpy(dtype='datetime64[ns]').astype(np.int64)
                day = times // DAY_NS * DAY_NS
                di = np.searchsorted(days, day)
                si = stocks.get_indexer(frame.instrument.astype(str))
                minute = times // MINUTE_NS % 1440
                slot = np.searchsorted(minutes, minute)
                valid = (di < len(days)) & (si >= 0) & (slot < bars) & (times % MINUTE_NS == 0)
                valid &= days[di.clip(0, len(days) - 1)] == day
                valid &= minutes[slot.clip(0, bars - 1)] == minute
                positions = np.flatnonzero(valid)
                block_ids = mapping[di[positions], si[positions]]
                positions, block_ids = positions[block_ids > 0], block_ids[block_ids > 0]
                local_slots = slot[positions]
                rows = block_ids.astype(np.int64) * bars + local_slots
                if len(np.unique(rows)) != len(rows) or seen[rows].any():
                    raise ValueError('原始价格中心存在重复网格位置')
                anchors[block_ids, local_slots] = anchor_view(frame.iloc[positions])
                seen[rows] = True
            anchors.flush()
            if report:
                report('price_memory_frequency_built', cache_split=split, frequency=f, observed_bars=int(seen.sum()))
        finally:
            anchors._mmap.close()
    return {path.name: old_coordinate._record_array(path) for path in sorted(staging.glob('*.npy'))}


def build_reference_arrays(store, parent_directory, parent, manifest, staging, report, split):
    """独立重建仅写三份 FP32 中心，不要求构建数量旁路。"""
    days = base._load_resident_array(parent_directory / 'days.npy', parent['files']['days.npy'])
    days = days.astype('datetime64[ns]').astype(np.int64)
    stocks = pd.Index(parent['stocks'])
    sources = {item['path'] for item in base_data_manifest(manifest)['files']}
    months = [str(month) for month in pd.DatetimeIndex(days).to_period('M').unique()]
    for f in FREQUENCIES:
        blocks, bars, width = parent['files'][f + '-values.npy']['shape']
        if width != 72 or bars != DAILY_BARS[f]:
            raise ValueError('父缓存频率布局不一致')
        mapping = base._load_resident_array(parent_directory / (f + '-map.npy'), parent['files'][f + '-map.npy'])
        anchors = np.lib.format.open_memmap(staging / (f + '-reference-values.npy'), mode='w+', dtype=np.float32, shape=(blocks, bars, 3))
        try:
            anchors[:] = np.nan
            seen = np.zeros(blocks * bars, bool)
            step = int(f[3:-1])
            minutes = np.r_[np.arange(570 + step, 691, step), np.arange(780 + step, 901, step)]
            available_months = [m for m in months if f'raw/cpt_jyc_2026_stock_{f}/{m}.parquet' in sources]
            for frame in raw_batches(store.data_root, f, available_months, pd.Timestamp(days[0]),
                                     pd.Timestamp(days[-1]) + pd.Timedelta(days=1), stocks=stocks,
                                     batch_rows=store.config['training']['resident_cache']['build_batch_rows']):
                times = frame.date.to_numpy(dtype='datetime64[ns]').astype(np.int64)
                day = times // DAY_NS * DAY_NS
                di = np.searchsorted(days, day)
                si = stocks.get_indexer(frame.instrument.astype(str))
                minute = times // MINUTE_NS % 1440
                slot = np.searchsorted(minutes, minute)
                valid = (di < len(days)) & (si >= 0) & (slot < bars) & (times % MINUTE_NS == 0)
                valid &= days[di.clip(0, len(days) - 1)] == day
                valid &= minutes[slot.clip(0, bars - 1)] == minute
                positions = np.flatnonzero(valid)
                block_ids = mapping[di[positions], si[positions]]
                positions, block_ids = positions[block_ids > 0], block_ids[block_ids > 0]
                local_slots = slot[positions]
                rows = block_ids.astype(np.int64) * bars + local_slots
                if len(np.unique(rows)) != len(rows) or seen[rows].any():
                    raise ValueError('原始价格中心存在重复网格位置')
                anchors[block_ids, local_slots] = reference_view(frame.iloc[positions])
                seen[rows] = True
            anchors.flush()
            if report:
                report('trade_reference_frequency_built', cache_split=split, frequency=f, observed_bars=int(seen.sum()))
        finally:
            anchors._mmap.close()
    return {path.name: old_coordinate._record_array(path) for path in sorted(staging.glob('*-reference-values.npy'))}


def ensure_side_cache(store, parent_directory, parent_key, manifest, *, split, report=None):
    parent_directory = Path(parent_directory)
    parent = read_json(parent_directory / 'manifest.json')
    if parent.get('format') != base.FORMAT or parent.get('status') != 'complete' or parent.get('cache_key') != parent_key:
        raise ValueError('基础输入缓存身份不一致')
    key = side_key(parent_key)
    directory = store.training_run.parent / 'trade-reference-cache' / key
    fingerprints = {name: item['sha256'] for name, item in parent['files'].items()}
    if directory.exists():
        saved = read_json(directory / 'manifest.json')
        if saved.get('status') != 'complete' or saved.get('format') != FORMAT or saved.get('cache_key') != key or saved.get('base_files') != fingerprints or saved.get('contract') != inputs.CONTRACT:
            raise ValueError('价格历史旁路身份改变或未完成')
        if any(saved.get(name) != parent.get(name) for name in ('stocks', 'catalog', 'frequencies')):
            raise ValueError('价格历史基础元数据改变')
        return directory, key
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=directory.name + '.building-', dir=directory.parent))
    try:
        found = reusable_anchor_source(store, parent)
        if found is None:
            files = build_anchor_arrays(store, parent_directory, parent, manifest, staging, report, split)
            anchor_directory = directory
            provenance = {'kind': 'rebuilt_raw_price_only', 'quantity_arrays_read': False}
            new_bytes = sum(v['bytes'] for v in files.values())
        else:
            anchor_directory, files, provenance = found
            new_bytes = 0
        reference_files = build_reference_arrays(store, parent_directory, parent, manifest, staging, report, split)
        files.update(reference_files)
        new_bytes += sum(v['bytes'] for v in reference_files.values())
        anchor_bytes = sum(v['array_bytes'] for v in files.values())
        saved = {'format': FORMAT, 'cache_key': key, 'status': 'complete', 'split': split,
            'base_cache_key': parent_key, 'base_directory': str(parent_directory.resolve()),
            'base_manifest_sha256_at_build': sha256(parent_directory / 'manifest.json'),
            'base_files': fingerprints, 'contract': inputs.CONTRACT, 'files': files,
            'anchor_directory': str(anchor_directory.resolve()), 'anchor_provenance': provenance,
            'catalog': parent['catalog'], 'input_precision': PRECISION,
            'stocks': parent['stocks'], 'frequencies': parent['frequencies'],
            'resident_array_bytes': parent['resident_array_bytes'] + anchor_bytes,
            'base_retained_channels': 72, 'side_array_bytes': anchor_bytes, 'new_array_disk_bytes': new_bytes,
            'label_values_read_by_side_builder': False, 'parent_cache_modified': False}
        write_json(staging / 'manifest.json', saved)
        if staging.resolve().parent != directory.parent.resolve() or directory.resolve().parent != directory.parent.resolve():
            raise ValueError('缓存发布路径超出预期目录')
        os.replace(staging, directory)
        return directory, key
    except BaseException:
        if staging.exists():
            old_coordinate._remove_own_staging(staging, directory)
        raise


class ResidentBlock:
    def __init__(self, directory, key, reserve_gib=2.0, report=None, *, resident=True):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / 'manifest.json')
        m = self.manifest
        if m.get('status') != 'complete' or m.get('format') != FORMAT or m.get('cache_key') != key or m.get('contract') != inputs.CONTRACT or side_key(m['base_cache_key']) != key:
            raise ValueError('价格历史缓存身份不一致')
        parent_dir = Path(m['base_directory'])
        parent = read_json(parent_dir / 'manifest.json')
        if parent['cache_key'] != m['base_cache_key'] or {n:r['sha256'] for n,r in parent['files'].items()} != m['base_files']:
            raise ValueError('价格历史缓存的父数组改变')
        if any(m.get(name) != parent.get(name) for name in ('stocks', 'catalog', 'frequencies')):
            raise ValueError('价格历史基础元数据改变')
        expected = {f + suffix: parent['files'][f + '-values.npy']['shape'][:2] + tail
                    for f in FREQUENCIES for suffix, tail in (('-anchor-values.npy', []), ('-reference-values.npy', [3]))}
        if set(m['files']) != set(expected):
            raise ValueError('成交参考旁路必须包含完整的价格中心及三字段数组')
        for name, shape in expected.items():
            if m['files'][name]['shape'] != shape or m['files'][name]['dtype'] != '<f4':
                raise ValueError('成交参考数组必须保持约定的FP32布局')
        if m['resident_array_bytes'] != parent['resident_array_bytes'] + sum(r['array_bytes'] for r in m['files'].values()):
            raise ValueError('成交参考数组资源记录不一致')
        numeric_names = {f + '-values.npy' for f in FREQUENCIES}
        metadata = sum(v['array_bytes'] for n,v in parent['files'].items() if n not in numeric_names)
        require_memory((m['resident_array_bytes'] if resident else metadata) + m['catalog']['rows'] * 8, reserve_gib)
        self._mapped = []
        arrays = {}
        try:
            for name, record in parent['files'].items():
                path = old_coordinate._safe_array_path(parent_dir, name)
                if not resident and name in numeric_names:
                    value = old_coordinate._mmap_verified(path, record)
                    self._mapped.append(value)
                else:
                    value = base._load_resident_array(path, record)
                arrays[name[:-4]] = value
            for name, record in m['files'].items():
                if name not in {f + suffix for f in FREQUENCIES for suffix in ('-anchor-values.npy', '-reference-values.npy')}:
                    raise ValueError('成交参考旁路包含未声明数组')
                path = old_coordinate._safe_array_path(Path(m['anchor_directory']) if name.endswith('-anchor-values.npy') else self.directory, name)
                if resident:
                    value = base._load_resident_array(path, record)
                else:
                    value = old_coordinate._mmap_verified(path, record)
                    self._mapped.append(value)
                arrays[name[:-4]] = value
            self.arrays = arrays
            self.targets, self.clock = arrays['targets'], arrays['sample_clock']
            self.day, self.stock = arrays['sample_day'], arrays['sample_stock']
            self.day_plans = {f: np.arange(len(arrays['days']))[:, None] + np.arange(-PREVIOUS_DAYS[f], 1) for f in FREQUENCIES}
            if report:
                report('price_memory_arrays_loaded', resident=resident, array_bytes=m['resident_array_bytes'])
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in self._mapped:
            value._mmap.close()
        self._mapped.clear()

    def training_targets(self):
        return self.targets, {stamp: value['reason'] for stamp, value in self.manifest['catalog']['skipped'].items()}

    def raw_batch(self, indices, destinations=None):
        indices = np.asarray(indices)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self.targets)).any():
            raise ValueError('非法样本索引')
        parent_destinations = None if destinations is None else {f: pair[0] for f, pair in destinations.items()}
        codes, repairs, clock = base.ResidentBlock.raw_batch(self, indices, parent_destinations)
        raw = {}
        for f in FREQUENCIES:
            days = self.day_plans[f][self.day[indices]]
            block = self.arrays[f + '-map'][days.clip(0), self.stock[indices, None]]
            block = np.where(days >= 0, block, 0)
            length = days.shape[1] * DAILY_BARS[f]
            anchors = (np.empty((len(indices), length), np.float32) if destinations is None
                       else destinations[f][1][:len(indices)])
            np.take(self.arrays[f + '-anchor-values'], block.reshape(-1), axis=0,
                    out=anchors.reshape(-1, DAILY_BARS[f]), mode='clip')
            references = (np.empty((len(indices), length, 3), np.float32) if destinations is None
                          else destinations[f][2][:len(indices)])
            np.take(self.arrays[f + '-reference-values'], block.reshape(-1), axis=0,
                    out=references.reshape(-1, DAILY_BARS[f], 3), mode='clip')
            raw[f] = codes[f], anchors, references
        return raw, repairs, clock

    def batch(self, indices):
        raw, repairs, clock = self.raw_batch(indices)
        raw = {f: tuple(torch.from_numpy(value) for value in pair) for f, pair in raw.items()}
        fixed = {f: tuple(torch.from_numpy(value) for value in pair) for f, pair in repairs.items()}
        windows = prepare_trade_reference_windows(raw, fixed, torch.from_numpy(clock))
        return {f: value.numpy() for f, value in windows.items()}, clock


class ResidentBatches:
    def __init__(self, block, order, targets, start, batch_size, device):
        self.block, self.order, self.targets = block, order, targets
        self.start, self.batch_size, self.device = start, batch_size, torch.device(device)

    def __iter__(self):
        cuda = self.device.type == "cuda"
        stream = torch.cuda.Stream(device=self.device) if cuda else None
        slots = []
        lengths = {f: (PREVIOUS_DAYS[f] + 1) * DAILY_BARS[f] for f in FREQUENCIES}
        for _ in range(2):
            tensors = {f: (torch.empty(self.batch_size, length, 72, dtype=torch.float16, pin_memory=cuda),
                           torch.empty(self.batch_size, length, dtype=torch.float32, pin_memory=cuda),
                           torch.empty(self.batch_size, length, 3, dtype=torch.float32, pin_memory=cuda)) for f,length in lengths.items()}
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
                return prepare_trade_reference_windows(values, fixed, clocks), clocks, target
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
        "base_cache_modified": False, "retained_base_channels": 72, "additional_side_array_bytes": block.manifest["side_array_bytes"],
        "new_array_disk_bytes": block.manifest["new_array_disk_bytes"], "anchor_provenance": block.manifest["anchor_provenance"],
        "load_and_build_seconds": time.perf_counter() - started}
    path = Path(run) / "resident-cache.json"
    if path.exists() and read_json(path)["cache_key"] != key:
        raise ValueError("恢复时坐标缓存身份改变")
    write_json(path, record)
    return ResidentStore(store, block, spec["evaluation_days"])
