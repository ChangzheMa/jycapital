"""全量常驻缓存的数值、因果性、全局排列、双缓冲和恢复验证。"""

import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd

from test_research_branches import ROOT, VERSIONS, MODULES, PROTOCOL, small_config
import test_research_pipeline as pipeline_tests
import test_research_entrypoints as entrypoint_tests
import resident_data as resident


def fixture(directory, version="v101", mini=False):
    config = small_config(version, resident=True)
    config["training"]["batch_size"] = 16
    protocol = copy.deepcopy(PROTOCOL)
    protocol["min_cross_section_size"] = 3
    protocol["splits"] = {
        "train": {"start": "2020-01-01", "stop": "2020-03-01"},
        "validation": {"start": "2020-03-01", "stop": "2020-04-01"},
        "test": {"start": "2020-04-01", "stop": "2020-05-01"}}
    training_days = (pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03"]) if mini else
                     pd.bdate_range("2020-01-02", periods=39).delete([10, 11, 12, 13]))
    days = training_days.append(pd.to_datetime(["2020-03-02", "2020-03-03", "2020-04-01"]))
    rows = [(day + pd.Timedelta(minutes=int(minute)), stock)
            for index, day in enumerate(days) for minute in resident.CLOCK_MINUTES
            for stock in ("A", "B", "C" if index < len(training_days) // 2 else "D")]
    keys = pd.DataFrame(rows, columns=["date", "instrument"])
    data, train, holdout = (directory / part for part in ("data", "train", "holdout"))
    data.mkdir(parents=True)
    keys.to_parquet(data / (protocol["universe_table"] + ".parquet"), index=False)
    rng = np.random.default_rng(731)
    labels = keys.copy()
    labels["vwap_return"] = rng.normal(0, .001, len(labels))
    labels.loc[labels.date == keys.date.iloc[0], "vwap_return"] = .1
    labels.loc[labels.index[4], "vwap_return"] = np.nan
    labels.to_parquet(data / (protocol["label_table"] + ".parquet"), index=False)
    module = MODULES[version]["data"]
    for frequency in resident.FREQUENCIES:
        times = pd.to_datetime(module.regular_times(days, frequency))
        frame = pd.DataFrame([(time, stock) for time in times for stock in ("A", "B", "C", "D")],
                             columns=["date", "instrument"])
        values = rng.normal(size=(len(frame), 72)).astype(np.float32)
        values[0] = 0
        values[4, 0] = 1e8
        values[8, 40] = -1e10
        values[12] = np.nan
        values[16, 1:] = np.nan
        values[16, 0] = 0
        frame = pd.concat([frame, pd.DataFrame(values, columns=resident.VALUE_COLUMNS)], axis=1)
        frame = frame.drop(index=frame.index[21:25])
        for month, part in frame.groupby(frame.date.dt.to_period("M")):
            split = "train" if str(month) < "2020-03" else "validation" if str(month) == "2020-03" else "test"
            path = (train / "data" if split == "train" else holdout / "data" / split) / frequency / (str(month) + ".parquet")
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(path, index=False)
    store = module.ProcessedStore(train, holdout, data, config, protocol)
    return store, config


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.store, cls.config = fixture(cls.root)
        cls.plan = resident.TrainingPlan(cls.store)
        cls.directory = cls.root / "cache"
        cls.key = "synthetic-source"
        cls.manifest = resident.build_cache(cls.store, cls.directory, cls.key,
                                            MODULES["v101"]["train"].prepare_training_targets, plan=cls.plan)
        cls.block = resident.ResidentBlock(cls.directory, cls.key, reserve_gib=0)

    @classmethod
    def tearDownClass(cls):
        del cls.block
        cls.temporary.cleanup()

    def reference(self, indices, version="v101"):
        module = MODULES[version]["data"]
        config = small_config(version, resident=True)
        store = module.ProcessedStore(self.store.training_run, self.store.holdout_run,
                                      self.store.data_root, config, self.store.protocol)
        chosen = self.plan.keys.iloc[indices]
        windows = {f: np.empty((len(indices), 75, length), np.float32)
                   for f, length in config["data"]["window_bars"].items()}
        clock = np.empty((len(indices), 8), np.float32)
        months = chosen.date.dt.to_period("M").astype(str).to_numpy()
        for month in np.unique(months):
            where = np.flatnonzero(months == month)
            block = store.load("train", month)
            key_index = pd.MultiIndex.from_frame(block.keys[["date", "instrument"]])
            selected = key_index.get_indexer(pd.MultiIndex.from_frame(chosen.iloc[where]))
            values, times = block.batch(selected)
            for f in windows:
                windows[f][where] = values[f]
            clock[where] = times
        return windows, clock

    def test_all_cutoffs_calendar_changes_partial_missing_and_overflow_match(self):
        indices = np.r_[np.arange(48), np.arange(len(self.plan.keys) - 48, len(self.plan.keys))]
        actual, clock = self.block.batch(indices)
        for version in VERSIONS:
            with self.subTest(version=version):
                expected, expected_clock = self.reference(indices, version)
                np.testing.assert_array_equal(clock, expected_clock)
                for f in resident.FREQUENCIES:
                    np.testing.assert_array_equal(actual[f], expected[f])
        self.assertTrue(any(item["fp32_exceptions"] > 0 for item in self.manifest["frequencies"].values()))
        self.assertEqual(self.block.arrays["bar1m-values"].dtype, np.float16)
        self.assertFalse(self.block.arrays["bar1m-values"].flags.writeable)

    def test_targets_are_original_full_section_targets_and_global_shuffle_once(self):
        pieces = [self.store.load("train", month).keys for month in self.store.months("train")]
        keys = pd.concat(pieces, ignore_index=True)
        trainer = MODULES["v101"]["train"]
        expected, skipped = trainer.prepare_training_targets(keys, self.store.protocol)
        actual, actual_skipped = self.block.training_targets()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_skipped, skipped)
        order = trainer.shuffled_sample_indices(actual, 42, 1, "global")
        np.testing.assert_array_equal(np.sort(order), np.flatnonzero(np.isfinite(actual)))
        np.testing.assert_array_equal(order, trainer.shuffled_sample_indices(actual, 42, 1, "global"))
        self.assertFalse(np.array_equal(order, trainer.shuffled_sample_indices(actual, 42, 2, "global")))
        self.assertGreater(self.plan.keys.iloc[order[:64]].date.dt.month.nunique(), 1)

    def test_block_deduplication_and_no_future_value_reaches_input(self):
        self.assertLess(self.plan.counts["bar1m"], len(self.plan.keys) * 2)
        index = np.array([len(self.plan.keys) - 24])  # 最后一天 09:30。
        actual, _ = self.block.batch(index)
        self.assertTrue((actual["bar1m"][:, :, 240:] == 0).all())
        self.assertTrue((self.block.batch([0])[0]["bar5m"] == 0).all())

    def test_double_buffer_tail_and_resume_cursor_cpu(self):
        trainer = MODULES["v101"]["train"]
        order = trainer.shuffled_sample_indices(self.block.targets, 42, 1, "global")[:53]
        seen = []
        for offset, selected, windows, clock, target in resident.ResidentBatches(
                self.block, order, self.block.targets, 14, 7, "cpu"):
            expected, expected_clock = self.block.batch(selected)
            for f in windows:
                np.testing.assert_array_equal(windows[f].numpy(), expected[f])
            np.testing.assert_array_equal(clock.numpy(), expected_clock)
            np.testing.assert_array_equal(target.numpy(), self.block.targets[selected])
            self.assertEqual(offset, 14 + len(seen))
            seen.extend(selected.tolist())
        np.testing.assert_array_equal(seen, order[14:])

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_double_buffer_cuda_reuse_and_sparse_fp32_repair(self):
        order = np.r_[np.arange(48), np.arange(len(self.plan.keys) - 48, len(self.plan.keys))]
        targets = np.nan_to_num(self.block.targets)
        for _, selected, windows, clock, target in resident.ResidentBatches(self.block, order, targets, 0, 7, "cuda"):
            expected, expected_clock = self.block.batch(selected)
            for f in windows:
                np.testing.assert_array_equal(windows[f].cpu().numpy(), expected[f])
            np.testing.assert_array_equal(clock.cpu().numpy(), expected_clock)
            np.testing.assert_array_equal(target.cpu().numpy(), targets[selected])

    def test_memory_budget_source_binding_and_corruption_rejected(self):
        with patch.object(resident.psutil, "virtual_memory", return_value=type("Memory", (), {"available": 0})()):
            with self.assertRaises(MemoryError):
                resident.ResidentBlock(self.directory, self.key, reserve_gib=0)
        with self.assertRaises(ValueError):
            resident.ResidentBlock(self.directory, "different-source", reserve_gib=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npy"
            np.save(path, np.arange(20, dtype=np.float16))
            record = {"bytes": path.stat().st_size, "shape": [20], "dtype": "<f2", "sha256": resident.sha256(path)}
            content = bytearray(path.read_bytes())
            content[-1] ^= 1
            path.write_bytes(content)
            with self.assertRaises(ValueError):
                resident._load_resident_array(path, record)

    def test_cache_reuse_and_bounded_validation(self):
        reused = resident.build_cache(self.store, self.directory, self.key,
                                     MODULES["v101"]["train"].prepare_training_targets)
        self.assertEqual(reused, self.manifest)
        store = resident.ResidentStore(self.store, self.block, evaluation_days=1)
        self.assertEqual(store.months("train"), ["global"])
        frames = [store.load("validation", segment).keys for segment in store.months("validation")]
        pd.testing.assert_frame_equal(pd.concat(frames, ignore_index=True), self.store.load("validation", "2020-03").keys)

    def test_cache_key_is_shared_and_binds_sources_and_target_policy(self):
        manifest = {"files": [{"path": "train/bar1m/test.parquet", "sha256": "original"}]}
        key = resident.cache_key(self.store, manifest, MODULES["v101"]["train"].prepare_training_targets)
        for version in VERSIONS:
            original = copy.copy(self.store)
            original.config = small_config(version, resident=True)
            self.assertEqual(key, resident.cache_key(original, manifest, MODULES[version]["train"].prepare_training_targets))
        changed = {"files": [{"path": "train/bar1m/test.parquet", "sha256": "changed"}]}
        self.assertNotEqual(key, resident.cache_key(self.store, changed, MODULES["v101"]["train"].prepare_training_targets))
        original = copy.copy(self.store)
        original.protocol = {**self.store.protocol, "min_cross_section_size": 30}
        self.assertNotEqual(key, resident.cache_key(original, manifest, MODULES["v101"]["train"].prepare_training_targets))

    def test_build_refuses_insufficient_disk_before_creating_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "cache"
            with patch.object(resident.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
                with self.assertRaises(OSError):
                    resident.build_cache(self.store, target, self.key,
                                         MODULES["v101"]["train"].prepare_training_targets, plan=self.plan)
            self.assertFalse(target.exists())


class GlobalTrainerTests(unittest.TestCase):
    def test_global_resume_is_exact_in_all_three_models(self):
        self.check_resume("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_global_resume_is_exact_in_all_three_models(self):
        self.check_resume("cuda")

    def check_resume(self, device):
        for version in VERSIONS:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base, config = fixture(root, version, mini=True)
                trainer = MODULES[version]["train"]
                resident.build_cache(base, root / "cache", "test-source", trainer.prepare_training_targets)
                block = resident.ResidentBlock(root / "cache", "test-source", reserve_gib=0)
                store = resident.ResidentStore(base, block, evaluation_days=1)
                configure, step = trainer.configure, trainer.training_step

                def chosen_device(spec):
                    configure(spec)
                    return torch.device(device)

                runs = [root / name for name in ("continuous", "resumed")]
                manifests = [{"status": "running"}, {"status": "running"}]
                for run in runs:
                    run.mkdir()
                    trainer.write_json(run / "data-manifest.json", {"synthetic": True})
                    trainer.write_json(run / "baseline-preservation.json", {})
                with patch.object(trainer, "configure", chosen_device), \
                        patch.object(trainer, "environment", lambda: {"packages": {}}), \
                        patch.object(trainer, "draw_history"), patch.object(trainer, "log_event"):
                    trainer.train(store, runs[0], config, manifests[0])
                    calls = 0

                    def interrupt(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise KeyboardInterrupt
                        return step(*args, **kwargs)

                    with patch.object(trainer, "training_step", interrupt), self.assertRaises(KeyboardInterrupt):
                        trainer.train(store, runs[1], config, manifests[1])
                    trainer.train(store, runs[1], config, manifests[1], resume=True)
                for filename in ("last.pt", "best.pt"):
                    left, right = [torch.load(run / filename, map_location="cpu", weights_only=True) for run in runs]
                    for name in left["state_dict"]:
                        torch.testing.assert_close(left["state_dict"][name], right["state_dict"][name], rtol=0, atol=0)
                frames = [pd.read_csv(run / "history.csv") for run in runs]
                columns = ["train_loss", "validation_loss", "ic_mean", "rank_ic_mean", "best_epoch", "stale_epochs", "patience_epochs"]
                pd.testing.assert_frame_equal(frames[0][columns], frames[1][columns], check_exact=True)
                self.assertTrue(all(item["status"] == "complete" for item in manifests))


class GlobalPatchPipelineTests(pipeline_tests.PipelineChecks, unittest.TestCase):
    version = "v101"
    resident = True


class GlobalMixerPipelineTests(pipeline_tests.PipelineChecks, unittest.TestCase):
    version = "v201"
    resident = True


class GlobalGRUPipelineTests(pipeline_tests.PipelineChecks, unittest.TestCase):
    version = "v301"
    resident = True


class GlobalEntrypoints(entrypoint_tests.EntrypointTests):
    config_factory = staticmethod(lambda version: small_config(version, resident=True))


if __name__ == "__main__":
    unittest.main()
