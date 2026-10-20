"""验证缓存跨边界历史、完整键、双缓冲数值及限定源码升级。"""

import contextlib
import copy
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd

from test_research_branches import ROOT, MODULES, create_model, small_config
from test_resident_data import fixture
from artifacts import read_json, write_json, sha256
import resident_data as resident
import evaluation_cache as cache
import evaluation_upgrade as upgrade


class EvaluationCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.original, cls.config = fixture(cls.root, "v201", mini=True)
        # 将极值、部分缺失及合法零值放入实际验证窗口，而非无关历史。
        for f in resident.FREQUENCIES:
            p = cls.original.path(f, "2020-03")
            frame = pd.read_parquet(p)
            frame.loc[0, resident.VALUE_COLUMNS[0]] = 1e8
            frame.loc[1, resident.VALUE_COLUMNS[40]] = -1e10
            frame.loc[2, resident.VALUE_COLUMNS] = np.nan
            frame.loc[3, resident.VALUE_COLUMNS] = 0
            frame.to_parquet(p, index=False)
        cls.run_dir = cls.root / "run"
        write_json(cls.run_dir / "data-manifest.json", {"synthetic": "evaluation-cache"})
        cls.identity = read_json(cls.run_dir / "data-manifest.json")
        cls.key = cache.cache_key(cls.original, "validation", cls.identity)
        cls.directory = cls.root / "evaluation-cache" / cls.key
        cache.build_evaluation_cache(cls.original, "validation", cls.directory, cls.key)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_resident_and_stream_match_all_cutoffs_history_masks_and_overflow(self):
        reference = self.original.load("validation", "2020-03")
        for mode in ("resident", "stream"):
            with self.subTest(mode=mode):
                data = cache.EvaluationData(self.directory, self.key, mode, 0, 7)
                try:
                    pd.testing.assert_frame_equal(data.keys, reference.keys)
                    for order in (np.arange(24), np.arange(24, 48)):
                        block = data.load_segment(order)
                        actual, clock = block.batch(order)
                        expected, expected_clock = reference.batch(order)
                        np.testing.assert_array_equal(clock, expected_clock)
                        for f in resident.FREQUENCIES:
                            np.testing.assert_array_equal(actual[f], expected[f])
                        if mode == "stream":
                            self.assertLessEqual(block.arrays["bar1m-values"].nbytes,
                                                 data.mapped["bar1m-values"].nbytes)
                finally:
                    data.close()

    def check_predictions(self, device):
        for version in ("v201", "v301"):
            trainer = MODULES[version]["train"]
            config = small_config(version, resident=True)
            config["inference"]["batch_size"] = 7
            original = MODULES[version]["data"].ProcessedStore(
                self.original.training_run, self.original.holdout_run, self.original.data_root,
                config, self.original.protocol)
            store = resident.ResidentStore(original, None, 1)
            model = create_model(version, config).to(device)
            for mode in ("resident", "stream"):
                store.evaluation_cache_mode = mode
                with self.subTest(version=version, mode=mode, device=device), contextlib.redirect_stdout(io.StringIO()):
                    expected, keys, scores = trainer.predict_split(store, "validation", model, device, config)
                    rng = torch.get_rng_state().clone()
                    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                    actual, actual_keys, actual_scores = trainer.predict_split(
                        store, "validation", model, device, config, self.run_dir)
                    pd.testing.assert_frame_equal(keys, actual_keys)
                    np.testing.assert_array_equal(actual.factor, expected.factor)
                    self.assertEqual(actual_scores[0], scores[0])
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []):
                        self.assertTrue(torch.equal(before, after))
                    with patch.object(original, "load_range", side_effect=AssertionError("不应重读原始网格")), \
                         patch.object(cache, "build_evaluation_cache", side_effect=AssertionError("不应重复构建")):
                        again, _, _ = trainer.predict_split(store, "validation", model, device, config, self.run_dir)
                    np.testing.assert_array_equal(again.factor, expected.factor)
                store.evaluation_cache[1].close()
                store.evaluation_cache = None

    def test_cpu_predictions_metrics_rng_and_cache_reuse_for_both_models(self):
        self.check_predictions(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_predictions_metrics_rng_and_buffer_reuse_for_both_models(self):
        self.check_predictions(torch.device("cuda"))

    def test_test_split_keeps_validation_history_and_all_invalid_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            original, config = fixture(Path(temp), "v301", mini=True)
            path = original.data_root / (original.protocol["label_table"] + ".parquet")
            labels = pd.read_parquet(path)
            labels.loc[labels.date >= "2020-04-01", original.protocol["label_column"]] = np.nan
            labels.to_parquet(path, index=False)
            directory = Path(temp) / "cache"
            cache.build_evaluation_cache(original, "test", directory, "test-invalid-labels")
            data = cache.EvaluationData(directory, "test-invalid-labels", "stream", 0, 7)
            try:
                reference = original.load("test", "2020-04")
                order = np.arange(len(data.keys))
                block = data.load_segment(order)
                actual, clock = block.batch(order)
                expected, expected_clock = reference.batch(order)
                pd.testing.assert_frame_equal(data.keys, reference.keys)
                self.assertTrue(data.keys.vwap_return.isna().all())
                np.testing.assert_array_equal(clock, expected_clock)
                for f in resident.FREQUENCIES:
                    np.testing.assert_array_equal(actual[f], expected[f])
            finally:
                data.close()

    def test_auto_memory_choice_and_forced_resident_guard(self):
        manifest = read_json(self.directory / "data/manifest.json")
        required = manifest["resident_array_bytes"]
        available = required // 2 + 2 * 7 * 880 * 72 * 2
        with patch.object(cache.psutil, "virtual_memory") as memory:
            memory.return_value.available = available
            data = cache.EvaluationData(self.directory, self.key, "auto", 0, 7)
            self.assertEqual(data.mode, "stream")
            data.close()
            with self.assertRaises(MemoryError):
                cache.EvaluationData(self.directory, self.key, "resident", 0, 7)

    def test_fingerprint_shared_across_models_but_bound_to_split_and_data(self):
        other = copy.copy(self.original)
        other.config = small_config("v301", resident=True)
        self.assertEqual(cache.cache_key(other, "validation", self.identity), self.key)
        self.assertNotEqual(cache.cache_key(other, "test", self.identity), self.key)
        self.assertNotEqual(cache.cache_key(other, "validation", {"changed": True}), self.key)

    def test_corrupt_numeric_cache_rejected_in_both_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "cache"
            shutil.copytree(self.directory, dest)
            path = dest / "data/bar1m-values.npy"
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                last = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([last[0] ^ 1]))
            for mode in ("stream", "resident"):
                with self.assertRaisesRegex(ValueError, "校验失败"):
                    cache.EvaluationData(dest, self.key, mode, 0, 7)


def legacy_trainer(source):
    """还原本次唯一允许的入口补丁，供已登记旧版本的升级测试使用。"""
    source = source.replace("from evaluation_cache import predict_cached_split\n", "")
    source = source.replace("from evaluation_upgrade import upgrade_evaluation_sources\n", "")
    source = source.replace("    if isinstance(store, ResidentStore) and run is not None:\n"
                            "        return predict_cached_split(store, split, model, device, config, run)\n", "")
    start = source.index('    parser.add_argument("--upgrade-evaluation"')
    stop = source.index('    parser.add_argument("--reproduction-only"', start)
    source = source[:start] + source[stop:]
    source = source.replace('    if args.upgrade_evaluation and not args.resume:\n'
                            '        parser.error("--upgrade-evaluation 必须与 --resume 一起使用")\n', "")
    source = source.replace('        if args.upgrade_evaluation:\n'
                            '            manifest = upgrade_evaluation_sources(ROOT, run, manifest)\n', "")
    return source.replace('        store.evaluation_cache_mode = args.evaluation_cache_mode\n', "")


class UpgradeTests(unittest.TestCase):
    def test_entrypoint_can_resume_legacy_checkpoint_with_explicit_upgrade(self):
        for version in ("v201", "v301"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                original, config = fixture(root, version, mini=True)
                trainer = MODULES[version]["train"]
                config["training"]["epochs"] = 1
                config["inference"]["batch_size"] = 7
                write_json(root / "config.json", config)
                write_json(root / "protocol.json", original.protocol)
                run = root / "run"
                argv = [version + "_train.py", "--config", str(root / "config.json"),
                        "--protocol", str(root / "protocol.json"), "--run-dir", str(run),
                        "--training-run", str(original.training_run), "--holdout-run", str(original.holdout_run),
                        "--data-root", str(original.data_root), "--training-confirmed", "--reproduction-only"]
                configure, step = trainer.configure, trainer.training_step
                calls = 0

                def cpu(spec):
                    configure(spec)
                    return torch.device("cpu")

                def interrupt(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise KeyboardInterrupt
                    return step(*args, **kwargs)

                with contextlib.redirect_stdout(io.StringIO()), patch.object(trainer, "configure", cpu), \
                        patch.object(trainer, "environment", lambda: {"packages": {}}), \
                        patch.object(trainer, "draw_history"), \
                        patch.object(trainer, "data_manifest", lambda store, path: write_json(
                            path / "data-manifest.json", {"synthetic": True})):
                    with patch.object(sys, "argv", argv), patch.object(trainer, "training_step", interrupt), \
                            self.assertRaises(KeyboardInterrupt):
                        trainer.main()
                    before = sha256(run / "last.pt")
                    checkpoint = torch.load(run / "last.pt", weights_only=True, map_location="cpu")
                    self.assertEqual(checkpoint["progress"]["optimizer_steps"], 2)
                    relative = f"solutions/{version}/{version}_train.py"
                    frozen = run / "source" / relative
                    frozen.write_text(legacy_trainer(frozen.read_text(encoding="utf-8")), encoding="utf-8")
                    manifest = read_json(run / "manifest.json")
                    manifest["source"]["files"][relative] = sha256(frozen)
                    write_json(run / "manifest.json", manifest)
                    with patch.object(sys, "argv", [*argv, "--resume", "--upgrade-evaluation"]):
                        trainer.main()
                self.assertEqual(read_json(run / "manifest.json")["status"], "complete")
                record = read_json(run / "source-upgrades/evaluation-cache-v1/upgrade.json")
                self.assertEqual(record["preserved"]["last.pt"], before)
                final = torch.load(run / "last.pt", weights_only=True, map_location="cpu")
                self.assertGreater(final["progress"]["optimizer_steps"], 2)
                self.assertEqual(final["progress"]["epoch"], 2)

    def setup_run(self, root, version):
        project, run = root / "project", root / "run"
        trainer = f"solutions/{version}/{version}_train.py"
        files = (trainer, f"solutions/{version}/{version}_model.py", "common/resident_data.py")
        for relative in files:
            current = project / relative
            frozen = run / "source" / relative
            current.parent.mkdir(parents=True, exist_ok=True)
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, current)
            if relative == trainer:
                frozen.write_text(legacy_trainer(current.read_text(encoding="utf-8")), encoding="utf-8")
            else:
                shutil.copy2(current, frozen)
        for name in ("evaluation_cache.py", "evaluation_upgrade.py"):
            shutil.copy2(ROOT / "common" / name, project / "common" / name)
        write_json(run / "config.json", small_config(version, resident=True))
        for name in ("protocol.json", "structural-sections.json", "data-manifest.json"):
            write_json(run / name, {"test": name})
        torch.save({"optimizer": {"step": 37}, "progress": {"epoch": 4, "sample_cursor": 512}}, run / "last.pt")
        torch.save({"weight": torch.tensor([1.0])}, run / "best.pt")
        manifest = {"solution_id": version, "status": "interrupted", "source": {
            "files": {name: sha256(run / "source" / name) for name in files}}}
        write_json(run / "manifest.json", manifest)
        return project, run, manifest

    def test_upgrade_preserves_checkpoint_config_and_original_source_for_both_models(self):
        for version in ("v201", "v301"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                project, run, manifest = self.setup_run(Path(temp), version)
                protected = {p.name: sha256(p) for p in run.iterdir() if p.is_file() and p.name != "manifest.json"}
                old = copy.deepcopy(manifest)
                updated = upgrade.upgrade_evaluation_sources(project, run, manifest)
                self.assertEqual(updated["evaluation_upgrade"]["kind"], "evaluation_cache_v1")
                for name, digest in protected.items():
                    self.assertEqual(sha256(run / name), digest)
                archive = run / "source-upgrades/evaluation-cache-v1"
                self.assertEqual(read_json(archive / "manifest-before.json"), old)
                for name, digest in updated["source"]["files"].items():
                    self.assertEqual(sha256(run / "source" / name), digest)
                self.assertEqual(upgrade.upgrade_evaluation_sources(project, run, updated), updated)

    def test_upgrade_rejects_training_model_or_unknown_entrypoint_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            project, run, manifest = self.setup_run(Path(temp), "v201")
            model = project / "solutions/v201/v201_model.py"
            model.write_bytes(model.read_bytes() + b'\n# changed\n')
            with self.assertRaisesRegex(ValueError, "其他已有源码"):
                upgrade.validate_upgrade(project, run, manifest)
            shutil.copy2(ROOT / "solutions/v201/v201_model.py", model)
            trainer = project / "solutions/v201/v201_train.py"
            trainer.write_bytes(trainer.read_bytes() + b'\nunauthorized_training_change = True\n')
            with self.assertRaisesRegex(ValueError, "已审核"):
                upgrade.validate_upgrade(project, run, manifest)
            self.assertFalse((run / "source-upgrades").exists())


if __name__ == "__main__":
    unittest.main()
