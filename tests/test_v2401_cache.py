"""坐标旁路的无标签构建、紧凑读取、流式读取及损坏拒绝。"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

from test_v2401 import small_config
import geometry_resident as base
from price_memory_inputs import FREQUENCIES, DAILY_BARS, PRICE_FIELDS
from coordinate_inputs import QUANTITY_FIELDS, transform_side
from price_memory_resident import ensure_side_cache, ResidentBlock, side_key, ResidentBatches
from v2401_data import WindowBlock, regular_times


def record_array(path, values):
    np.save(path, values, allow_pickle=False)
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(values.shape), "dtype": values.dtype.str, "array_bytes": values.nbytes}


def fixture(root):
    config = small_config(resident=True)
    training, data, parent = root / "training", root / "data", root / "base-cache"
    training.mkdir(); data.mkdir(); parent.mkdir()
    days = pd.to_datetime(["2020-01-02", "2020-01-03"])
    stocks = ["S0", "S1"]
    minutes = (570, 600, 630, 660, 690, 810, 840, 870)
    keys = pd.DataFrame([(day+pd.Timedelta(minutes=m), stock) for day in days for m in minutes for stock in stocks],
                        columns=["date", "instrument"])
    arrays = {"days": days.to_numpy(dtype="datetime64[ns]").astype(np.int64),
              "sample_day": np.repeat(np.arange(2,dtype=np.int32),16),
              "sample_stock": np.tile(np.arange(2,dtype=np.int32),16),
              "sample_clock": np.tile(np.repeat(np.arange(8,dtype=np.uint8),2),2),
              "targets": np.linspace(-1,1,len(keys))}
    grids, raw_frames, files, sources, parameters = {}, {}, {}, [], {}
    rng = np.random.default_rng(2011)
    for frequency in FREQUENCIES:
        bars = DAILY_BARS[frequency]
        times = regular_times(days, frequency)
        frame = pd.DataFrame([(pd.Timestamp(t), stock) for t in times for stock in stocks], columns=["date", "instrument"])
        for n,field in enumerate(PRICE_FIELDS): frame[field] = 10 + n*.01 + np.arange(len(frame))*.001
        for n,field in enumerate(QUANTITY_FIELDS): frame[field] = (np.arange(len(frame)) + n) % 31
        directory = data / ("cpt_jyc_2026_stock_"+frequency); directory.mkdir()
        path = directory / "2020-01.parquet"
        frame.to_parquet(path,index=False)
        raw_frames[frequency] = path
        sources.append({"path": "raw/"+path.relative_to(data).as_posix()})
        parameters[frequency] = {"book_volume":{"center":0.,"scale":1e-5},
                                 "book_order_count":{"center":1.,"scale":2.},
                                 "price_anchor":{"center":2.,"scale":1.}}
        quantity, anchor = transform_side(frame, parameters[frequency])
        full = rng.normal(size=(5,bars,72)).astype(np.float32)
        full[0] = np.nan
        full[1,0,0] = 100000.
        full[1,0,60] = 90000.  # 父数量通道必须保留，极值也须精确恢复。
        with np.errstate(over="ignore",invalid="ignore"): half = full.astype(np.float16)
        positions = np.flatnonzero(np.isfinite(full) & ~np.isfinite(half))
        block_ids = positions // (bars*72)
        arrays[frequency+"-values"] = half
        arrays[frequency+"-map"] = np.array([[1,3],[2,4]],np.int32)
        arrays[frequency+"-exception-pointers"] = np.r_[0,np.bincount(block_ids,minlength=5).cumsum()].astype(np.int64)
        arrays[frequency+"-exception-offsets"] = (positions % (bars*72)).astype(np.int32)
        arrays[frequency+"-exception-values"] = full.reshape(-1)[positions]
        native = np.full((len(times),2,73),np.nan,np.float32)
        for day in range(2):
            for stock in range(2):
                block = 1 + stock*2 + day
                native[day*bars:(day+1)*bars,stock,:72] = full[block,:,:72]
        native[...,72] = anchor.reshape(len(times),2)
        grids[frequency] = times,native
    for name,values in arrays.items(): files[name+".npy"] = record_array(parent/(name+".npy"),values)
    key = "f"*64
    manifest = {"format":base.FORMAT,"status":"complete","cache_key":key,"files":files,
        "stocks":stocks,"frequencies":list(FREQUENCIES),"catalog":{"rows":len(keys),"skipped":{}},
        "resident_array_bytes":sum(v["array_bytes"] for v in files.values())}
    (parent/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    store = SimpleNamespace(training_run=training,data_root=data,config=config,parameters=parameters)
    native = WindowBlock(keys,grids,stocks,config,days.to_numpy())
    return store,parent,key,{"files":sources},native,raw_frames


class CoordinateCacheTests(unittest.TestCase):
    def test_no_label_build_compact_and_stream_match_native_and_preserve_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,parent,key,manifest,native,_ = fixture(Path(tmp))
            preserved = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir()}
            directory,new_key = ensure_side_cache(store,parent,key,manifest,split="train")
            self.assertFalse((store.data_root/"cpt_jyc_2026_vwap.parquet").exists())
            rows = np.arange(len(native.keys))
            expected,clocks = native.batch(rows)
            for resident in (True,False):
                block = ResidentBlock(directory,new_key,reserve_gib=2.0,resident=resident)
                try:
                    actual,actual_clocks = block.batch(rows)
                    np.testing.assert_array_equal(clocks,actual_clocks)
                    raw, repairs, parent_clock = block.raw_batch(rows)
                    original = base.prepare_device_windows(
                        {f: torch.from_numpy(pair[0]) for f, pair in raw.items()},
                        {f: tuple(torch.from_numpy(v) for v in pair) for f, pair in repairs.items()},
                        torch.from_numpy(parent_clock))
                    for frequency in FREQUENCIES:
                        np.testing.assert_array_equal(actual[frequency],expected[frequency])
                        np.testing.assert_array_equal(actual[frequency][:, :75], original[frequency].numpy())
                        self.assertEqual(block.arrays[frequency+"-values"].shape[-1],72)
                    self.assertFalse(block.manifest["label_values_read_by_side_builder"])
                finally: block.close()
            self.assertEqual(preserved,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir()})

    def test_prefetch_order_and_rng_match_direct_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,parent,key,manifest,_,_ = fixture(Path(tmp))
            directory,key = ensure_side_cache(store,parent,key,manifest,split="train")
            block = ResidentBlock(directory,key,2.0)
            order = np.arange(len(block.targets))[::-1].copy()
            state = torch.get_rng_state().clone()
            batches = list(ResidentBatches(block,order,block.targets,3,5,"cpu"))
            torch.testing.assert_close(state,torch.get_rng_state(),atol=0,rtol=0)
            for offset,indices,windows,clock,target in batches:
                expected,expected_clock = block.batch(order[offset:offset+5])
                np.testing.assert_array_equal(indices,order[offset:offset+5])
                for f in FREQUENCIES: np.testing.assert_array_equal(windows[f].numpy(),expected[f])
                np.testing.assert_array_equal(clock.numpy(),expected_clock)
                np.testing.assert_array_equal(target.numpy(),block.targets[indices])

    def test_duplicate_raw_positions_are_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,parent,key,manifest,_,paths = fixture(Path(tmp))
            path = paths["bar1m"]
            frame = pd.read_parquet(path)
            pd.concat([frame,frame.iloc[:1]],ignore_index=True).to_parquet(path,index=False)
            with self.assertRaisesRegex(ValueError,"重复"):
                ensure_side_cache(store,parent,key,manifest,split="train")
            self.assertFalse((store.training_run.parent/"price-memory-cache"/side_key(key)).exists())

    def test_modified_payload_and_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store,parent,key,manifest,_,_ = fixture(Path(tmp))
            directory,key = ensure_side_cache(store,parent,key,manifest,split="train")
            path = directory/"manifest.json"
            original = path.read_bytes();metadata=json.loads(original)
            metadata["contract"]["distance"] = "changed_contract"
            path.write_text(json.dumps(metadata),encoding="utf-8")
            with self.assertRaises(ValueError): ResidentBlock(directory,key,2.0)
            path.write_bytes(original)
            payload = directory/"bar1m-anchor-values.npy"
            with payload.open("r+b") as stream:
                stream.seek(-1,2);old=stream.read(1);stream.seek(-1,2);stream.write(bytes([old[0]^1]))
            with self.assertRaises(ValueError): ResidentBlock(directory,key,2.0,resident=False)


    def test_existing_anchor_reuse_reads_no_quantity_and_rejects_mutation(self):
        from unittest.mock import patch
        import coordinate_resident as old
        import price_memory_resident as current
        with tempfile.TemporaryDirectory() as tmp:
            store,parent,key,manifest,native,_=fixture(Path(tmp))
            old_directory,_=old.ensure_side_cache(store,parent,key,manifest,split='train')
            original_load=current.base._load_resident_array
            observed=[]
            def checked_load(path,record):
                observed.append(Path(path).name)
                self.assertNotIn('quantity',Path(path).name)
                return original_load(path,record)
            with patch.object(current,'raw_batches',side_effect=AssertionError('reused anchors must not query raw data')):
                directory,new_key=current.ensure_side_cache(store,parent,key,manifest,split='train')
            saved=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['new_array_disk_bytes'],0)
            self.assertEqual(Path(saved['anchor_directory']),old_directory.resolve())
            self.assertFalse(list(directory.glob('*.npy')))
            with patch.object(current.base,'_load_resident_array',side_effect=checked_load):
                block=current.ResidentBlock(directory,new_key,2.)
            try:
                expected,clock=native.batch(np.arange(len(native.keys)))
                actual,actual_clock=block.batch(np.arange(len(native.keys)))
                for f in FREQUENCIES:np.testing.assert_array_equal(actual[f],expected[f])
                np.testing.assert_array_equal(actual_clock,clock)
            finally:block.close()
            payload=old_directory/'bar1m-anchor-values.npy'
            with payload.open('r+b') as stream:
                stream.seek(-1,2);value=stream.read(1);stream.seek(-1,2);stream.write(bytes([value[0]^1]))
            with self.assertRaises(ValueError):current.ResidentBlock(directory,new_key,2.,resident=False)


if __name__ == "__main__":
    unittest.main()
