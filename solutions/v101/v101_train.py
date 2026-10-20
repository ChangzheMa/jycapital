"""v101 单次正式训练、验证选择、冻结后测试。行情缓存不进入提交包。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Windows CUDA DLL 必须先于 NumPy/OpenMP 加载。
import torch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT / "common", ROOT / "evaluation"):
    sys.path.insert(0, str(directory))
from artifacts import environment, experiment_filename, log_event, now, read_json, sha256, snapshot_sources, write_json
from early_stopping import patience_epochs, select_epoch, should_stop, validate_early_stopping
from baseline_evaluation import evaluate
from v101_data import ProcessedStore, KEYS, FREQUENCIES
from v101_model import MultiFrequencyPatchTransformer, validate_model_config
from v101_data import section_indices
from v101_loss import standardized_huber_loss, standardized_targets, sample_huber_loss
from protocol_compatibility import validate_protocol_compatibility
from resident_data import (ResidentBatches, ResidentStore, TrainingPlan, open_resident_store, validate_resident_config)
from training_migration import (prepare_state, remaining_order, validate_implementation, verify_source, state_digest,
                                source_record, validate_transition)
from training_resources import record_resources


def atomic_torch_save(value, path):
    started = time.perf_counter()
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    log_event(Path(path).parent, "checkpoint_saved", checkpoint=Path(path).name,
              seconds=round(time.perf_counter() - started, 6))


def configure(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(config["training"]["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_batch(block, index, device):
    windows, clock = block.batch(index)
    return {f: torch.from_numpy(x).to(device) for f, x in windows.items()}, torch.from_numpy(clock).to(device)


def blocks(store, split, months):
    """单线程预取下一月份，避免 Windows 多进程复制大网格。"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(store.load, split, months[0])
        for i, month in enumerate(months):
            block = pending.result()
            if i + 1 < len(months):
                pending = pool.submit(store.load, split, months[i + 1])
            yield month, block
            del block


def predict_split(store, split, model, device, config, run=None):
    model.eval()
    frames, labels = [], []
    batch = config["inference"]["batch_size"]
    with torch.inference_mode():
        for month, block in blocks(store, split, store.months(split)):
            values = np.empty(len(block.keys), dtype=np.float64)
            for offset in range(0, len(values), batch):
                index = np.arange(offset, min(offset + batch, len(values)))
                windows, clock = tensor_batch(block, index, device)
                values[index] = (model(windows, clock) / config["inference"]["factor_scale"]).cpu().numpy()
            frame = block.keys[KEYS].copy()
            frame["factor"] = values
            frames.append(frame)
            labels.append(block.keys)
            if run:
                log_event(run, "predicted_month", split=split, month=month, rows=len(values))
            del block
    predictions, keys = pd.concat(frames, ignore_index=True), pd.concat(labels, ignore_index=True)
    return predictions, keys, evaluate(predictions, keys, store.protocol)


def save_evaluation(run, split, predictions, result):
    stats, sections, monthly = result
    directory = run / "evaluation" / split
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics.json", stats)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    sections.to_parquet(directory / "cross-sections.parquet", index=False)
    monthly.to_csv(directory / "monthly.csv", index=False, encoding="utf-8-sig")


def data_manifest(store, run):
    files = []
    for table in (store.protocol["universe_table"], store.protocol["label_table"]):
        path = store.data_root / (table + ".parquet")
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    references = {}
    for name, source in (("train", store.training_run), ("holdout", store.holdout_run)):
        manifest = read_json(source / "manifest.json")
        if manifest["status"] != "complete":
            raise ValueError(f"未完成的预处理来源: {source}")
        changes = validate_protocol_compatibility(
            read_json(source / "protocol.json"), read_json(run / "protocol.json"),
            allow_selection_metric_change=True)
        references[name] = {"path": str(source.resolve()), "manifest_sha256": sha256(source / "manifest.json"),
                            "protocol_sha256": sha256(source / "protocol.json"), "protocol_differences": changes}
        summary = read_json(source / "summary.json")
        for frequency in FREQUENCIES:
            normalization = source / "normalization" / f"{frequency}.json"
            if name == "train" and sha256(normalization) != summary["frequencies"][frequency]["statistics_sha256"]:
                raise ValueError("训练归一化参数与已完成预处理记录不同")
            target = run / "normalization" / normalization.name
            target.parent.mkdir(exist_ok=True)
            if name == "train":
                shutil.copyfile(normalization, target)
            if sha256(normalization) != sha256(target):
                raise ValueError("训练与留出集的归一化参数不同")
        for path in sorted((source / "data").rglob("*.parquet")):
            relative = path.relative_to(source / "data")
            if relative.parts[-2] not in FREQUENCIES:
                continue
            record = read_json(source / "records" / relative.with_suffix(".json"))
            digest = sha256(path)
            if path.stat().st_size != record["bytes"] or digest != record["sha256"]:
                raise ValueError(f"预处理分区已改变: {path}")
            files.append({"path": name + "/" + relative.as_posix(), "bytes": record["bytes"], "sha256": digest})
        log_event(run, "verified_preprocessing", source=name)
    result = {"mode": "local", "root": str(store.data_root.resolve()), "files": files, "preprocessing": references}
    write_json(run / "data-manifest.json", result)


def draw_history(run, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("train_loss", "validation_loss"):
        axes[0].plot(frame.epoch, frame[key], label=key)
    for key in ("ic_mean", "rank_ic_mean"):
        axes[1].plot(frame.epoch, frame[key], label="validation_" + key)
    for axis in axes:
        axis.set_xlabel("Epoch")
        if (run / "migration.json").exists():
            boundary = read_json(run / "migration.json")["sample_handoff"]["epoch"] - .5
            axis.axvline(boundary, color="gray", linestyle="--", label="FP16/global migration")
        axis.legend()
    fig.tight_layout()
    fig.savefig(run / "learning-curves.png", dpi=160)
    plt.close(fig)


def validate_training(config, protocol):
    validate_model_config(config)
    spec = config["training"]
    validate_resident_config(config)
    expected = {"loss": "cross_section_zscore_huber",
                "loss_reduction": "mean_samples", "loss_precision": "float64",
                "target_transform": "cross_section_zscore", "label_ddof": 0,
                "structural_policy": "finite_nonconstant_labels",
                "constant_prediction_policy": "allow_huber_gradient"}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("v101 必须先按完整截面标准化标签，再使用已登记的样本 Huber 配置")
    if any(key in spec for key in ("target_scale", "target_clip", "sections_per_step", "checkpoint_sections")):
        raise ValueError("v101 不裁剪或固定缩放标签，不接受逐截面更新配置")
    if not math.isfinite(spec.get("huber_delta", float("nan"))) or spec["huber_delta"] <= 0:
        raise ValueError("Huber delta 必须为有限正数")
    if protocol["selection_metric"] != "ic_mean" or config["inference"]["factor_scale"] != 1:
        raise ValueError("v101 按验证 IC 选择，输出保留标准化目标单位")
    validate_early_stopping(spec)
    for key in ("batch_size", "checkpoint_batches", "epochs"):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] < 1:
            raise ValueError("batch 大小、训练次数与 checkpoint 间隔必须为正整数")


def structural_catalog(store, run):
    path = run / "structural-sections.json"
    if path.exists():
        saved = read_json(path)
        if saved["data_manifest_sha256"] != sha256(run / "data-manifest.json"):
            raise ValueError("结构性退化清单的数据指纹不同")
        if isinstance(store, ResidentStore) and saved.get("resident_cache_key") != store.block.manifest["cache_key"]:
            raise ValueError("续跑的常驻缓存已改变")
        return saved
    result = {"policy": "finite_nonconstant_labels",
              "data_manifest_sha256": sha256(run / "data-manifest.json"), "months": {}}
    if isinstance(store, ResidentStore):
        result["months"]["global"] = store.block.manifest["catalog"]
        result["resident_cache_key"] = store.block.manifest["cache_key"]
        write_json(path, result)
        return result
    minimum = store.protocol["min_cross_section_size"]
    for month, block in blocks(store, "train", store.months("train")):
        groups = section_indices(block.keys)
        skipped, counts = {}, {}
        finite_rows = int(np.isfinite(block.keys[store.protocol["label_column"]]).sum())
        for indices in groups:
            keys = block.keys.iloc[indices]
            labels = keys[store.protocol["label_column"]].to_numpy(dtype=np.float64)
            eligible = np.isfinite(labels)
            if int(eligible.sum()) < minimum:
                reason = "insufficient_labels"
            elif np.ptp(labels[eligible]) == 0:
                reason = "constant_labels"
            else:
                reason = None
            if reason:
                stamp = pd.Timestamp(keys.date.iloc[0]).isoformat()
                skipped[stamp] = {"reason": reason, "rows": len(indices),
                                  "finite_labels": int(eligible.sum())}
                counts[reason] = counts.get(reason, 0) + 1
        result["months"][month] = {"sections": len(groups), "rows": len(block.keys),
                                    "finite_label_rows": finite_rows, "skipped": skipped}
        log_event(run, "prepared_sections", month=month, sections=len(groups),
                  finite_labels=finite_rows, skipped=counts)
        del block
    write_json(path, result)
    return result


def initial_progress():
    return {"epoch": 1, "month_cursor": 0, "sample_cursor": 0, "best": -float("inf"),
            "best_epoch": 0, "stale": 0, "history": [],
            "loss_sum": 0.0, "sections": 0, "rows": 0,
            "optimizer_steps": 0, "epoch_steps": 0, "skipped": {},
            "epoch_seconds": 0.0}


def prepare_training_targets(keys, protocol):
    """完整截面标准化先于样本打乱，目标始终与原始行位置对齐。"""
    targets = np.full(len(keys), np.nan, dtype=np.float64)
    skipped = {}
    for indices in section_indices(keys):
        labels = torch.from_numpy(keys.iloc[indices][protocol["label_column"]].to_numpy(dtype=np.float64))
        values, reason = standardized_targets(labels, protocol["min_cross_section_size"])
        if reason:
            stamp = pd.Timestamp(keys.date.iloc[indices[0]]).isoformat()
            skipped[stamp] = reason
        else:
            targets[indices] = values.numpy()
    return targets, skipped


def shuffled_sample_indices(targets, seed, epoch, month):
    indices = np.flatnonzero(np.isfinite(targets))
    partition_seed = 0 if month == "global" else int(month.replace("-", ""))
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, partition_seed]))
    generator.shuffle(indices)
    return indices


def restore_rng(checkpoint, device):
    torch.set_rng_state(checkpoint["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def training_batches(block, order, targets, start, batch_size, device):
    if hasattr(block, "raw_batch"):
        yield from ResidentBatches(block, order, targets, start, batch_size, device)
        return
    for offset in range(start, len(order), batch_size):
        indices = order[offset:offset + batch_size]
        windows, clock = tensor_batch(block, indices, device)
        target = torch.from_numpy(targets[indices]).to(device)
        yield offset, indices, windows, clock, target


def training_step(model, optimizer, windows, clock, targets, config, amp):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(clock.device.type, dtype=torch.bfloat16, enabled=amp):
        prediction = model(windows, clock)
    loss = sample_huber_loss(prediction, targets, config["training"]["huber_delta"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"],
                                   error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "rows": len(targets)}


def huber_loss_check(predictions, keys, protocol, config):
    frame = keys.merge(predictions, on=KEYS, how="left", validate="one_to_one")
    losses, model_undefined = [], 0
    for _, group in frame.groupby("date", sort=True):
        scores = torch.from_numpy(group.factor.to_numpy(dtype=np.float64) * config["inference"]["factor_scale"])
        labels = torch.from_numpy(group[protocol["label_column"]].to_numpy(dtype=np.float64))
        result = standardized_huber_loss(scores, labels, protocol["min_cross_section_size"],
                                         config["training"]["huber_delta"])
        if result.loss is not None:
            losses.append(result.loss.item())
            model_undefined += int(result.ic is None)
    if not losses:
        raise ValueError("没有有效评估 Huber 截面")
    return float(np.mean(losses)), model_undefined


def verify_parent(root, saved):
    for relative, expected in saved.items():
        if sha256(root / relative) != expected:
            raise ValueError("父版本冻结产物已改变: " + relative)


def parent_snapshot(root, config):
    if config["parent_run_id"] is None:
        # 代码父版本尚无训练运行时，保存其源码指纹，不虚构父权重。
        solution = root / "solutions" / config["parent_solution_id"]
        if not solution.is_dir():
            raise FileNotFoundError(solution)
        return {path.relative_to(root).as_posix(): sha256(path)
                for path in sorted(solution.iterdir())
                if path.is_file() and path.suffix in {".py", ".json", ".txt"}}
    parent = root / "outputs/runs" / config["parent_run_id"]
    parent_manifest = read_json(parent / "manifest.json")
    if parent_manifest["status"] != "complete" or sha256(parent / "best.pt") != parent_manifest["selection"]["model_sha256"]:
        raise ValueError("父版本冻结权重或状态异常")
    files = {}
    for directory in (parent, root / "solutions" / parent_manifest["solution_id"]):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".tmp"):
                files[path.relative_to(root).as_posix()] = sha256(path)
    parent_config = read_json(parent / "config.json")
    result_path = root / "experiments" / experiment_filename(
        parent_manifest["solution_id"], parent_config.get("model", {}).get("family"))
    if result_path.exists():
        files[result_path.relative_to(root).as_posix()] = sha256(result_path)
    path = root / "evaluation/baseline_evaluation.py"
    files[path.relative_to(root).as_posix()] = sha256(path)
    return files


def initialize_migration(store, run, config, manifest, catalog, model, optimizer, device):
    if not isinstance(store, ResidentStore):
        raise ValueError("迁移目标必须加载全局常驻缓存")
    source = Path(manifest["migration_source"]).resolve()
    validate_implementation(source, HERE)
    plan = TrainingPlan(store.original)
    state, record, last, best = prepare_state(source, run, config, store.protocol, catalog,
                                             plan.keys, store.block.targets, read_json(run / "data-manifest.json"))
    del plan
    shutil.copyfile(source / "history.csv", run / "legacy-history.csv")
    model.load_state_dict(best["state_dict"], strict=True)
    baseline_started = time.perf_counter()
    manifest.update(stage="migration_baseline_validation")
    write_json(run / "manifest.json", manifest)
    log_event(run, "migration_baseline_started", source_best_epoch=best["epoch"])
    predictions, keys, result = predict_split(store, "validation", model, device, config, run)
    huber, undefined = huber_loss_check(predictions, keys, store.protocol, config)
    if result[0]["ic_mean"] is None or undefined or result[0]["coverage_violations"]:
        raise ValueError("原最佳权重在新输入精度下未通过验证，不能初始化选模基线")
    atomic_torch_save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                       "config": config, "protocol": store.protocol, "epoch": best["epoch"]}, run / "best.pt")
    save_evaluation(run, "validation", predictions, result)
    save_evaluation(run / "migration-baseline", "validation", predictions, result)
    baseline = {"epoch": best["epoch"], "source_best_sha256": record["source_files"]["best.pt"],
                "new_best_sha256": sha256(run / "best.pt"), "metrics": result[0], "huber_loss": huber,
                "seconds": time.perf_counter() - baseline_started, "input_precision": config["data"]["input_precision"],
                "legacy_validation_ic": state["best"], "preserved_stale_epochs": state["stale"]}
    write_json(run / "migration-baseline.json", baseline)
    state["best"] = result[0]["ic_mean"]
    # 基线验证与继续优化分开；最后恢复 last.pt、完整 AdamW 和随机状态。
    model.load_state_dict(last["state_dict"], strict=True)
    optimizer.load_state_dict(last["optimizer"])
    if state_digest(model.state_dict()) != state_digest(last["state_dict"]) or state_digest(optimizer.state_dict()) != state_digest(last["optimizer"]):
        raise ValueError("迁移后的模型或优化器状态未精确继承")
    restore_rng(last, device)
    record.update(optimizer_inherited=True, rng_inherited=True,
                  last_state_sha256=state_digest(last["state_dict"]),
                  optimizer_state_sha256=state_digest(last["optimizer"]),
                  validation_baseline=baseline)
    write_json(run / "migration.json", record)
    verify_source(record)
    manifest["migration"] = {"kind": record["kind"], "source_run_id": record["source_run_id"],
                             "epoch": state["epoch"], "sample_handoff": record["sample_handoff"]}
    log_event(run, "migration_initialized", source_run_id=source.name, epoch=state["epoch"],
              processed_samples=record["sample_handoff"]["processed_samples"],
              remaining_samples=record["sample_handoff"]["remaining_samples"], optimizer_inherited=True)
    return state


def train(store, run, config, manifest, resume=False):
    validate_training(config, store.protocol)
    if (config["training"]["batching"] == "global_shuffled_samples") != isinstance(store, ResidentStore):
        raise ValueError("训练采样配置与数据加载器不一致")
    device = configure(config)
    spec = config["training"]
    catalog = structural_catalog(store, run)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = MultiFrequencyPatchTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"],
                                 weight_decay=spec["weight_decay"])
    env = environment()
    env.update(device=str(device), device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
               torch=torch.__version__, cuda=torch.version.cuda, training_autocast_bfloat16=amp,
               evaluation_precision="float32", loss_precision="float64",
               parameters=sum(p.numel() for p in model.parameters()), model_family=config["model"]["family"],
               recurrent_precision=config["model"].get("recurrent_precision"),
               input_precision=config["data"].get("input_precision", "float32"), batching=spec["batching"])
    write_json(run / "environment.json", env)
    (run / "requirements-lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in env["packages"].items()) + "\n",
                                               encoding="utf-8")
    state = initial_progress()
    if resume and (run / "last.pt").exists():
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["catalog_sha256"] != sha256(run / "structural-sections.json"):
            raise ValueError("续跑配置或结构性清单不同")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["progress"]
        restore_rng(checkpoint, device)
        if manifest.get("migration_source"):
            verify_source(read_json(run / "migration.json"))
    elif manifest.get("migration_source"):
        state = initialize_migration(store, run, config, manifest, catalog, model, optimizer, device)
    log_event(run, "training_started", parameters=env["parameters"], device=env["device_name"], resume=resume)
    manifest.update(status="running", stage="training")
    write_json(run / "manifest.json", manifest)
    started = checkpoint_clock = time.perf_counter()
    catalog_digest = sha256(run / "structural-sections.json")

    def save_progress():
        nonlocal checkpoint_clock
        current = time.perf_counter()
        state["epoch_seconds"] += current - checkpoint_clock
        checkpoint_clock = current
        atomic_torch_save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "config": config, "progress": state, "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                           "catalog_sha256": catalog_digest}, run / "last.pt")
        write_json(run / "progress.json", {k: state[k] for k in (
            "epoch", "month_cursor", "sample_cursor", "optimizer_steps", "best_epoch", "stale")})

    if manifest.get("migration_source") and not (run / "last.pt").exists():
        save_progress()
    for epoch in range(state["epoch"], spec["epochs"] + 1):
        if should_stop(state):
            break
        months = np.random.default_rng(config["seed"] + epoch).permutation(store.months("train")).tolist()
        remaining = months[state["month_cursor"]:]
        if remaining:
            for month, block in blocks(store, "train", remaining):
                month_start, month_batches = time.perf_counter(), 0
                targets, actual_skipped = (block.training_targets() if hasattr(block, "training_targets")
                                           else prepare_training_targets(block.keys, store.protocol))
                skipped = catalog["months"][month]["skipped"]
                if actual_skipped != {stamp: record["reason"] for stamp, record in skipped.items()}:
                    raise ValueError("标准化目标与预检退化截面清单不同")
                order = shuffled_sample_indices(targets, config["seed"], epoch, month)
                handoff_order = remaining_order(run, state, config["seed"], epoch)
                if handoff_order is not None:
                    order = handoff_order
                for offset, indices, windows, clock, target in training_batches(
                        block, order, targets, state["sample_cursor"], spec["batch_size"], device):
                    step = training_step(model, optimizer, windows, clock, target, config, amp)
                    state["loss_sum"] += step["loss"] * step["rows"]
                    state["rows"] += step["rows"]
                    state["optimizer_steps"] += 1
                    state["epoch_steps"] += 1
                    month_batches += 1
                    state["sample_cursor"] = offset + len(indices)
                    if state["epoch_steps"] % spec["checkpoint_batches"] == 0:
                        save_progress()
                if isinstance(store, ResidentStore):
                    state["skipped"] = {}
                for record in skipped.values():
                    reason = record["reason"]
                    state["skipped"][reason] = state["skipped"].get(reason, 0) + 1
                if isinstance(store, ResidentStore):
                    state["sections"] = catalog["months"][month]["sections"] - len(skipped)
                else:
                    state["sections"] += catalog["months"][month]["sections"] - len(skipped)
                state.update(month_cursor=state["month_cursor"] + 1, sample_cursor=0)
                save_progress()
                log_event(run, "trained_month", epoch=epoch, month=month, month_cursor=state["month_cursor"],
                          rows=len(order), batches_updated=month_batches,
                          optimizer_steps=state["optimizer_steps"],
                          seconds=round(time.perf_counter() - month_start, 3))
                del block
        validation_started = time.perf_counter()
        predictions, keys, result = predict_split(store, "validation", model, device, config, run)
        metric = result[0]["ic_mean"]
        validation_loss, model_undefined = huber_loss_check(predictions, keys, store.protocol, config)
        validation_seconds = time.perf_counter() - validation_started
        if metric is None:
            raise ValueError("验证 IC 无定义，不能选择权重")
        eligible = model_undefined == 0 and result[0]["coverage_violations"] == 0
        improved = select_epoch(state, metric, epoch, eligible=eligible)
        if improved:
            atomic_torch_save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                               "config": config, "protocol": store.protocol, "epoch": epoch}, run / "best.pt")
            save_evaluation(run, "validation", predictions, result)
        if state["rows"] == 0:
            raise ValueError("整轮没有可优化的样本")
        train_loss = state["loss_sum"] / state["rows"]
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
               "train_huber_loss": train_loss,
               "training_batching": spec["batching"], "batch_size": spec["batch_size"],
               "train_loss_reduction": spec["loss_reduction"],
               "validation_huber_loss": validation_loss, "validation_ic_mean": metric,
               "validation_rank_ic_mean": result[0]["rank_ic_mean"],
               "ic_mean": metric, "rank_ic_mean": result[0]["rank_ic_mean"],
               "best_epoch": state["best_epoch"], "improved": bool(improved),
               "selection_eligible": bool(eligible), "model_undefined_sections": model_undefined,
               "stale_epochs": state["stale"], "patience_epochs": patience_epochs(state["best_epoch"]),
               "train_rows": state["rows"],
               "train_sections": state["sections"], "optimizer_steps": state["epoch_steps"],
               "skipped_sections": sum(state["skipped"].values()),
               "validation_seconds": validation_seconds,
               "seconds": round(state["epoch_seconds"] + time.perf_counter() - checkpoint_clock, 3)}
        migration = state.get("migration")
        row["training_phase"] = "global_fp16" if isinstance(store, ResidentStore) else "monthly_fp32"
        if migration and epoch == migration["epoch"]:
            new_rows = state["rows"] - migration["legacy_rows"]
            if new_rows != migration["remaining_samples"]:
                raise ValueError("迁移轮次的剩余样本未恰好处理一次")
            row.update(training_phase="monthly_fp32_then_global_fp16",
                       legacy_train_rows=migration["legacy_rows"], new_train_rows=new_rows,
                       legacy_train_loss=migration["legacy_loss_sum"] / migration["legacy_rows"] if migration["legacy_rows"] else None,
                       new_train_loss=(state["loss_sum"] - migration["legacy_loss_sum"]) / new_rows if new_rows else None,
                       legacy_epoch_seconds=migration["legacy_seconds"],
                       new_epoch_seconds=row["seconds"] - migration["legacy_seconds"],
                       legacy_optimizer_steps=migration["legacy_epoch_steps"],
                       new_optimizer_steps=state["epoch_steps"] - migration["legacy_epoch_steps"])
        state["history"].append(row)
        pd.DataFrame(state["history"]).to_csv(run / "history.csv", index=False, encoding="utf-8-sig")
        log_event(run, "epoch_finished", **row)
        record_resources(run, device)
        state.update(epoch=epoch + 1, month_cursor=0, sample_cursor=0, loss_sum=0.0,
                     sections=0, rows=0, epoch_steps=0, skipped={}, epoch_seconds=0.0)
        checkpoint_clock = time.perf_counter()
        save_progress()
    if not (run / "best.pt").exists():
        raise ValueError("没有通过完整验证检查的 checkpoint")
    selection = {"best_epoch": state["best_epoch"], "best_validation_metric": state["best"],
                 "best_validation_ic": state["best"], "epochs_completed": len(state["history"]),
                 "selection_metric": "ic_mean", "patience_epochs": patience_epochs(state["best_epoch"]),
                 "early_stopping": dict(spec["early_stopping"]), "stale_epochs": state["stale"],
                 "model_sha256": sha256(run / "best.pt"),
                 "selected_at": now(), "optimizer_steps": state["optimizer_steps"],
                 "training_seconds_this_session": round(time.perf_counter() - started, 3),
                 "stop_reason": "early_stopping" if should_stop(state) else "max_epochs"}
    if manifest.get("migration_source"):
        baseline = read_json(run / "migration-baseline.json")
        selection.update(training_trajectory="monthly_fp32_then_global_fp16",
                         initialization_source_run=read_json(run / "migration.json")["source_run_id"],
                         selection_baseline_epoch=baseline["epoch"], selection_baseline_ic=baseline["metrics"]["ic_mean"])
    write_json(run / "selection.json", selection)
    manifest.update(stage="test_evaluation", selection=selection)
    write_json(run / "manifest.json", manifest)
    log_event(run, "model_frozen", **selection)
    draw_history(run, state["history"])
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_started = time.perf_counter()
    log_event(run, "test_evaluation_started")
    predictions, keys, result = predict_split(store, "test", model, device, config, run)
    test_loss, undefined = huber_loss_check(predictions, keys, store.protocol, config)
    log_event(run, "test_evaluation_finished", seconds=time.perf_counter() - test_started)
    save_evaluation(run, "test", predictions, result)
    preservation = read_json(run / "baseline-preservation.json")
    verify_parent(ROOT, preservation)
    write_json(run / "loss-consistency.json", {
        "loss": "cross_section_zscore_huber", "delta": spec["huber_delta"],
        "label_ddof": 0, "prediction_normalized": False, "ic_loss_identity_applicable": False,
        "training_batching": spec["batching"], "batch_size": spec["batch_size"],
        "training_loss_reduction": spec["loss_reduction"], "evaluation_loss_reduction": "equal_cross_sections",
        "selection_metric": "ic_mean", "factor_scale": config["inference"]["factor_scale"],
        "best_validation_huber_loss": (read_json(run / "migration-baseline.json")["huber_loss"]
             if manifest.get("migration_source") and state["best_epoch"] == read_json(run / "migration-baseline.json")["epoch"]
             else next(r["validation_huber_loss"] for r in state["history"] if r["epoch"] == state["best_epoch"])),
        "test_huber_loss": test_loss,
        "test_model_undefined_sections": undefined, "parent_preserved": bool(preservation),
    })
    manifest.update(status="complete", stage="complete", completed_at=now(), selection=selection,
                    metrics={s: read_json(run / "evaluation" / s / "metrics.json")
                             for s in ("validation", "test")})
    write_json(run / "manifest.json", manifest)
    if manifest.get("migration_source"):
        verify_source(read_json(run / "migration.json"))
    log_event(run, "completed", validation_ic=state["best"], test_ic=result[0]["ic_mean"],
              validation_rank_ic=manifest["metrics"]["validation"]["rank_ic_mean"],
              test_rank_ic=result[0]["rank_ic_mean"])
    record_resources(run, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "model-config.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "evaluation/protocols/local-v2-ic.json")
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--holdout-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resident-cache", type=Path, help="共享的全量 FP16 缓存目录")
    parser.add_argument("--migrate-from", type=Path, help="从月内 FP32 运行的 last.pt 显式迁移到新的全局运行")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reproduction-only", action="store_true",
                        help="仅供独立复现工作区使用，显式记录缺少历史比较产物")
    parser.add_argument("--training-confirmed", action="store_true",
                        help="仅在用户明确确认开始训练后使用")
    args = parser.parse_args()
    if not args.training_confirmed:
        parser.error("v101 训练尚待用户确认；确认后才可传入 --training-confirmed")
    if args.migrate_from and args.resume:
        parser.error("首次迁移用 --migrate-from；新迁移 run 的后续恢复只用 --resume")
    config, protocol, run = read_json(args.config), read_json(args.protocol), args.run_dir.resolve()
    validate_training(config, protocol)
    if args.migrate_from and (run == args.migrate_from.resolve() or run.is_relative_to(args.migrate_from.resolve())):
        parser.error("迁移必须使用独立新运行目录，不能写入来源运行")
    if args.migrate_from:
        provenance, _, _ = source_record(args.migrate_from)
        validate_transition(provenance["source_config"], config, provenance["source_protocol"], protocol)
        validate_implementation(args.migrate_from, HERE)
    if run.exists() and not args.resume:
        raise FileExistsError("运行目录已存在；续跑须显式指定 --resume")
    run.mkdir(parents=True, exist_ok=True)
    if args.resume:
        manifest = read_json(run / "manifest.json")
        if manifest["status"] == "complete":
            raise ValueError("已完成的运行不能续跑")
        if config != read_json(run / "config.json") or protocol != read_json(run / "protocol.json"):
            raise ValueError("续跑配置或协议已改变")
        for path, digest in manifest["source"]["files"].items():
            if sha256(ROOT / path) != digest:
                raise ValueError(f"续跑源码已改变: {path}")
    else:
        write_json(run / "config.json", config)
        write_json(run / "protocol.json", protocol)
        manifest = {"run_id": run.name, "solution_id": "v101", "status": "running", "created_at": now(),
                    "parent_solution_id": config["parent_solution_id"],
                    "parent_run_id": config["parent_run_id"], "references": config["references"],
                    "source": snapshot_sources(ROOT, run, solution_id="v101")}
        if args.migrate_from:
            manifest["migration_source"] = str(args.migrate_from.resolve())
        write_json(run / "manifest.json", manifest)
    try:
        if not args.resume:
            preservation_path = ROOT / "outputs/run-control" / run.name / "parent-preservation.json"
            if preservation_path.exists():
                preservation = read_json(preservation_path)
            elif args.reproduction_only:
                preservation = {}
            else:
                preservation = parent_snapshot(ROOT, config)
            verify_parent(ROOT, preservation)
            write_json(run / "baseline-preservation.json", preservation)
            manifest["baseline_verification"] = "protected" if preservation else "unavailable_reproduction"
            write_json(run / "manifest.json", manifest)
        else:
            verify_parent(ROOT, read_json(run / "baseline-preservation.json"))
        store = ProcessedStore(args.training_run, args.holdout_run, args.data_root, config, protocol)
        verification_started = time.perf_counter()
        data_manifest(store, run)
        log_event(run, "data_verification_finished", seconds=time.perf_counter() - verification_started)
        if config["training"]["batching"] == "global_shuffled_samples":
            manifest.update(stage="resident_cache")
            write_json(run / "manifest.json", manifest)
            store = open_resident_store(store, run, prepare_training_targets, args.resident_cache,
                                        lambda event, **fields: log_event(run, event, **fields))
        train(store, run, config, manifest, args.resume)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), stopped_at=now())
        write_json(run / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
