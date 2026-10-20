"""迁移衔接、完整优化器继承及新运行精确恢复；仅使用临时合成行情。"""

import contextlib
import copy
import hashlib
import io
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd

from test_resident_data import fixture
from test_research_branches import MODULES, ROOT
import resident_data as resident
import training_migration as migration
from artifacts import read_json, write_json, snapshot_sources


class MigrationTests(unittest.TestCase):
    def test_cli_does_not_write_into_source_or_combine_migration_with_resume(self):
        trainer = MODULES["v101"]["train"]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            marker = source / "original.txt"
            marker.write_text("preserved", encoding="utf-8")
            for target, extra in [(source, []), (Path(temporary) / "new", ["--resume"])]:
                arguments = ["v101_train.py", "--training-run", "unused", "--holdout-run", "unused",
                             "--run-dir", str(target), "--migrate-from", str(source), "--training-confirmed", *extra]
                with patch.object(sys, "argv", arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    trainer.main()
                self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")
                if target != source:
                    self.assertFalse(target.exists())

    def test_transition_only_allows_storage_and_sampling_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base, new = fixture(Path(temporary), mini=True)
            old = self.monthly(new)
            migration.validate_transition(old, new, base.protocol, base.protocol)
            for section, key, value in [("training", "learning_rate", .01), ("training", "epochs", 101),
                                        ("training", "batch_size", 8), ("model", "hidden_dim", 100)]:
                changed = copy.deepcopy(new)
                changed[section][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    migration.validate_transition(old, changed, base.protocol, base.protocol)

    @staticmethod
    def monthly(config):
        old = copy.deepcopy(config)
        old["data"].pop("input_precision")
        old["training"]["batching"] = "monthly_shuffled_samples"
        old["training"].pop("resident_cache")
        return old

    def test_migration_and_interrupted_resume_cpu(self):
        self.check_migration("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_migration_and_interrupted_resume_cuda(self):
        self.check_migration("cuda")

    def check_migration(self, device):
        trainer = MODULES["v101"]["train"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, config = fixture(root, mini=True)
            # 用可精确表示的常量截面，避免测试来源被旧标签中心化的舍入误差影响。
            labels_path = base.data_root / (base.protocol["label_table"] + ".parquet")
            labels = pd.read_parquet(labels_path)
            labels.loc[labels.date == labels.date.iloc[0], base.protocol["label_column"]] = .125
            labels.to_parquet(labels_path, index=False)
            config["training"].update(epochs=3, batch_size=7, checkpoint_batches=1)
            old_config = self.monthly(config)
            old_store = MODULES["v101"]["data"].ProcessedStore(base.training_run, base.holdout_run,
                                                                    base.data_root, old_config, base.protocol)
            old_batches = 0
            for month in old_store.months("train"):
                targets, _ = trainer.prepare_training_targets(old_store.load("train", month).keys, base.protocol)
                old_batches += (int(np.isfinite(targets).sum()) + 6) // 7
            source = root / "legacy"
            source.mkdir()
            write_json(source / "config.json", old_config)
            write_json(source / "protocol.json", base.protocol)
            write_json(source / "data-manifest.json", {"synthetic": True})
            write_json(source / "baseline-preservation.json", {})
            old_manifest = {"status": "running", "source": snapshot_sources(ROOT, source, solution_id="v101")}
            write_json(source / "manifest.json", old_manifest)
            configure, step = trainer.configure, trainer.training_step

            def chosen_device(spec):
                configure(spec)
                return torch.device(device)

            calls = 0
            def interrupt_legacy(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls > old_batches + 2:
                    raise KeyboardInterrupt
                return step(*args, **kwargs)

            with patch.object(trainer, "configure", chosen_device), patch.object(trainer, "environment", lambda: {"packages": {}}), \
                    patch.object(trainer, "draw_history"), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(trainer, "training_step", interrupt_legacy), self.assertRaises(KeyboardInterrupt):
                    trainer.train(old_store, source, old_config, old_manifest)
                old_manifest["status"] = "interrupted"
                write_json(source / "manifest.json", old_manifest)
                original_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
                checkpoint = torch.load(source / "last.pt", weights_only=True, map_location="cpu")
                self.assertEqual(checkpoint["progress"]["epoch"], 2)
                self.assertEqual(len(checkpoint["progress"]["history"]), 1)
                resident.build_cache(base, root / "cache", "migration-test", trainer.prepare_training_targets)
                block = resident.ResidentBlock(root / "cache", "migration-test", reserve_gib=0)
                store = resident.ResidentStore(base, block, evaluation_days=1)
                keys = resident.TrainingPlan(base).keys
                remaining, audit = migration.reconstruct_remaining(keys, block.targets, old_config,
                    checkpoint["progress"], read_json(source / "structural-sections.json"), base.protocol)
                self.assertEqual(audit["processed_samples"], checkpoint["progress"]["rows"])
                self.assertEqual(len(remaining) + audit["processed_samples"], int(np.isfinite(block.targets).sum()))
                self.assertTrue(audit["disjoint"] and audit["exhaustive"])
                # 独立按旧月内索引重建键集合，而不是用新的全局排列跳过行数。
                progress = checkpoint["progress"]
                months = np.random.default_rng(old_config["seed"] + progress["epoch"]).permutation(old_store.months("train"))
                processed_keys = []
                for cursor, month in enumerate(months[:progress["month_cursor"] + 1]):
                    part = old_store.load("train", month)
                    targets, _ = trainer.prepare_training_targets(part.keys, base.protocol)
                    order = trainer.shuffled_sample_indices(targets, old_config["seed"], progress["epoch"], month)
                    take = order if cursor < progress["month_cursor"] else order[:progress["sample_cursor"]]
                    processed_keys.extend(map(tuple, part.keys.iloc[take][["date", "instrument"]].itertuples(index=False, name=None)))
                expected = set(map(tuple, keys.iloc[np.flatnonzero(np.isfinite(block.targets))].itertuples(index=False, name=None))) - set(processed_keys)
                actual = set(map(tuple, keys.iloc[remaining].itertuples(index=False, name=None)))
                self.assertEqual(actual, expected)
                wrong = trainer.shuffled_sample_indices(block.targets, 42, progress["epoch"], "global")[progress["rows"]:]
                self.assertNotEqual(set(wrong), set(remaining))
                bad = copy.deepcopy(progress)
                bad["rows"] += 1
                with self.assertRaises(ValueError):
                    migration.reconstruct_remaining(keys, block.targets, old_config, bad,
                        read_json(source / "structural-sections.json"), base.protocol)

                targets = [root / name for name in ("continuous", "resumed")]
                manifests = []
                for run in targets:
                    run.mkdir()
                    write_json(run / "config.json", config)
                    write_json(run / "protocol.json", base.protocol)
                    write_json(run / "data-manifest.json", {"synthetic": True})
                    write_json(run / "baseline-preservation.json", {})
                    manifests.append({"status": "running", "migration_source": str(source)})
                checked_inheritance = []

                def inspect_initial_step(model, optimizer, *args, **kwargs):
                    if not checked_inheritance:
                        self.assertEqual(migration.state_digest(model.state_dict()), migration.state_digest(checkpoint["state_dict"]))
                        self.assertEqual(migration.state_digest(optimizer.state_dict()), migration.state_digest(checkpoint["optimizer"]))
                        baseline_weight = torch.load(targets[0] / "best.pt", weights_only=True, map_location="cpu")
                        source_best = torch.load(source / "best.pt", weights_only=True, map_location="cpu")
                        self.assertEqual(baseline_weight["config"], config)
                        self.assertEqual(migration.state_digest(baseline_weight["state_dict"]), migration.state_digest(source_best["state_dict"]))
                        baseline_report = read_json(targets[0] / "migration-baseline.json")
                        self.assertEqual(baseline_report["metrics"], read_json(targets[0] / "evaluation/validation/metrics.json"))
                        self.assertEqual(baseline_report["new_best_sha256"], hashlib.sha256((targets[0] / "best.pt").read_bytes()).hexdigest())
                        checked_inheritance.append(True)
                    return step(model, optimizer, *args, **kwargs)

                with patch.object(trainer, "training_step", inspect_initial_step):
                    trainer.train(store, targets[0], config, manifests[0])
                calls = 0
                def interrupt_new(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise KeyboardInterrupt
                    return step(*args, **kwargs)
                with patch.object(trainer, "training_step", interrupt_new), self.assertRaises(KeyboardInterrupt):
                    trainer.train(store, targets[1], config, manifests[1])
                trainer.train(store, targets[1], config, manifests[1], resume=True)
                left, right = [torch.load(run / "last.pt", weights_only=True, map_location="cpu") for run in targets]
                self.assertEqual(migration.state_digest(left["state_dict"]), migration.state_digest(right["state_dict"]))
                self.assertEqual(migration.state_digest(left["optimizer"]), migration.state_digest(right["optimizer"]))
                self.assertEqual(migration.state_digest(left["torch_rng"]), migration.state_digest(right["torch_rng"]))
                self.assertEqual(migration.state_digest(left["cuda_rng"]), migration.state_digest(right["cuda_rng"]))
                h1, h2 = [pd.read_csv(run / "history.csv") for run in targets]
                columns = ["epoch", "train_rows", "train_loss", "validation_loss", "ic_mean", "rank_ic_mean",
                           "best_epoch", "stale_epochs", "optimizer_steps", "training_phase", "new_train_rows"]
                pd.testing.assert_frame_equal(h1[columns], h2[columns], check_exact=True)
                mixed = h1[h1.epoch == 2].iloc[0]
                self.assertEqual(mixed.legacy_train_rows + mixed.new_train_rows, int(np.isfinite(block.targets).sum()))
                self.assertEqual(mixed.training_phase, "monthly_fp32_then_global_fp16")
                self.assertEqual(h1.loc[h1.epoch == 3, "train_rows"].iloc[0], int(np.isfinite(block.targets).sum()))
                report = read_json(targets[0] / "migration.json")
                self.assertTrue(report["optimizer_inherited"] and report["rng_inherited"])
                self.assertEqual(report["optimizer_state_sha256"], migration.state_digest(checkpoint["optimizer"]))
                baseline = read_json(targets[0] / "migration-baseline.json")
                self.assertEqual(baseline["epoch"], checkpoint["progress"]["best_epoch"])
                self.assertEqual(baseline["input_precision"], "fp16_with_fp32_overflow")
                for name, digest in original_hashes.items():
                    self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), digest, name)


if __name__ == "__main__":
    unittest.main()
