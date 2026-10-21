"""v802 单次正式训练、验证选择、冻结后测试。行情缓存不进入提交包。"""

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
from v802_data import ProcessedStore, KEYS, FREQUENCIES
from v802_model import MultiFrequencyGRU, validate_model_config
from v802_data import section_indices
from v802_loss import standardized_huber_loss, standardized_targets, sample_huber_loss, pearson_ic_loss
from v802_ema import WeightEMA
from protocol_compatibility import validate_protocol_compatibility
from resident_data import (ResidentBatches, ResidentStore, open_resident_store, validate_resident_config)
from training_resources import record_resources
from v802_initialization import initialize_backbone
from v802_sections import SectionStore, SectionBatches
from v802_evaluation import predict_cached_split
from v802_text_artifact import state_digest


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
    return predict_cached_split(store, split, model, device, config, run)


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
        axis.legend()
    fig.tight_layout()
    fig.savefig(run / "learning-curves.png", dpi=160)
    plt.close(fig)


def validate_training(config, protocol):
    validate_model_config(config)
    spec = config["training"]
    if spec["batching"] != "global_shuffled_cross_sections" or spec["loss"] != "zscore_huber_plus_cross_section_ic":
        raise ValueError("v802 要求完整截面与已登记组合损失")
    if spec["frozen_backbone"] is not False:
        raise ValueError("v802 要求联合训练时序主干")
    if spec["target_transform"] != "cross_section_zscore" or spec["label_ddof"] != 0:
        raise ValueError("原收益截面目标不可改变")
    if spec["loss_precision"] != "float64" or spec["loss_reduction"] != "mean_samples":
        raise ValueError("损失须使用 FP64 和截面内样本均值")
    validate_resident_config(dict(config, training=dict(spec, batching="global_shuffled_samples")))
    validate_early_stopping(spec)
    for name in ("huber_delta", "ic_auxiliary_weight", "learning_rate", "backbone_learning_rate", "gradient_clip"):
        if not math.isfinite(spec[name]) or spec[name] <= 0:
            raise ValueError("损失和训练尺度必须为有限正数")
    for name in ("batch_size", "checkpoint_batches", "epochs"):
        if isinstance(spec[name], bool) or not isinstance(spec[name], int) or spec[name] < 1:
            raise ValueError("批量记录和训练次数必须为正整数")
    averaging = spec["weight_averaging"]
    if averaging["method"] != "ema" or averaging["evaluation_weights"] != "ema":
        raise ValueError("v802 按 EMA 验证选模")
    if protocol["selection_metric"] != "ic_mean" or config["inference"]["factor_scale"] != 1:
        raise ValueError("v802 保持验证 IC 和原输出尺度")
    if config["initialization"]["method"] != "local_parent_backbone":
        raise ValueError("初始化必须来自已登记的本地 EMA 父模型")


def structural_catalog(store, run):
    block = store.block
    result = {"policy": "finite_nonconstant_labels", "data_manifest_sha256": sha256(run / "data-manifest.json"),
              "months": {"global": block.manifest["catalog"]}, "resident_cache_key": block.manifest["cache_key"]}
    path = run / "structural-sections.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("恢复时行情缓存或结构性清单改变")
    write_json(path, result)
    return result


def initial_progress():
    return {"epoch": 1, "month_cursor": 0, "sample_cursor": 0, "section_cursor": 0, "best": -float("inf"),
            "best_epoch": 0, "stale": 0, "history": [],
            "loss_sum": 0.0, "main_loss_sum": 0.0, "ic_loss_sum": 0.0, "sections": 0, "rows": 0,
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


def shuffled_sample_indices(block, targets, seed, epoch, month):
    order = np.array([number for number in range(len(block.boundaries) - 1)
                      if np.isfinite(targets[block.section(number)]).any()], dtype=np.int64)
    np.random.default_rng(np.random.SeedSequence([seed, epoch, 0])).shuffle(order)
    return order


def restore_rng(checkpoint, device):
    torch.set_rng_state(checkpoint["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def training_batches(block, order, targets, start, batch_size, device):
    yield from SectionBatches(block, order, targets, start, batch_size, device)


def training_step(model, optimizer, windows, clock, targets, config, amp):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(clock.device.type, dtype=torch.bfloat16, enabled=amp):
        prediction = model(windows, clock)
    valid = torch.isfinite(targets)
    if not valid.any() or not torch.isfinite(prediction).all():
        raise ValueError("完整截面没有有效目标或预测非有限")
    main_loss = sample_huber_loss(prediction[valid], targets[valid], config["training"]["huber_delta"])
    ic_loss = pearson_ic_loss(prediction, targets, 2).loss
    if ic_loss is None:
        raise ValueError("训练截面的 IC 损失无定义")
    loss = main_loss + config["training"]["ic_auxiliary_weight"] * ic_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "main_loss": float(main_loss.detach()),
            "ic_loss": float(ic_loss.detach()), "rows": int(valid.sum())}


def preflight_inputs(config, count, device):
    windows = {f: torch.randn(count, 75, length, device=device) for f, length in config["data"]["window_bars"].items()}
    for value in windows.values():
        value[:, 72:] = 1
    clock = torch.nn.functional.one_hot(torch.full((count,), 7, device=device), 8).float()
    return windows, clock


def preflight_encoder(model, config, device):
    count = config["inference"]["stock_day_batch_size"] if device.type == "cuda" else 2
    windows = {f: torch.randn(count, 75, length, device=device) for f, length in config["data"]["window_bars"].items()}
    for value in windows.values():
        value[:, 72:] = 1
    clock = torch.nn.functional.one_hot(torch.full((count,), 7, device=device), 8).float()
    with torch.inference_mode():
        features = model.encode_shared(windows, clock)
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise FloatingPointError("时序编码器的 FP32 快速路径不合法")


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
    # 保护父运行的冻结模型与评估证据，不锁定仍可追加的监控和分析记录。
    files = {}
    names = ("best.pt", "selection.json", "config.json", "protocol.json", "data-manifest.json")
    paths = [parent / name for name in names]
    for folder in (parent / "source", parent / "normalization", parent / "evaluation"):
        paths.extend(path for path in folder.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for path in paths:
        files[path.relative_to(root).as_posix()] = sha256(path)
    return files


def train(store, run, config, manifest, resume=False):
    validate_training(config, store.protocol)
    if not isinstance(store, SectionStore):
        raise ValueError("训练采样配置与数据加载器不一致")
    device = configure(config)
    spec = config["training"]
    catalog = structural_catalog(store, run)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = MultiFrequencyGRU(config).to(device)
    initial_backbone = initialize_backbone(model, config, store.backbone_artifact)
    averaging = spec["weight_averaging"]
    ema = WeightEMA(model, averaging["decay"], averaging["warmup_epochs"])
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": spec["backbone_learning_rate"], "name": "temporal_backbone"},
        {"params": model.context.parameters(), "lr": spec["learning_rate"], "name": "market_context"},
    ], weight_decay=spec["weight_decay"])
    env = environment()
    env.update(device=str(device), device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
               torch=torch.__version__, cuda=torch.version.cuda, training_autocast_bfloat16=amp,
               evaluation_precision="float32", loss_precision="float64",
               evaluation_weights="ema", ema_decay=averaging["decay"],
               ema_warmup_epochs=averaging["warmup_epochs"],
               parameters=sum(p.numel() for p in model.parameters()), model_family=config["model"]["family"],
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               frozen_backbone=False, gradient_checkpointing=config["model"]["gradient_checkpointing"],
               encoder_chunk_size=config["model"]["encoder_chunk_size"],
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
        if "section_cursor" not in state:
            raise ValueError("缺少完整截面恢复游标")
        if "ema_state_dict" not in checkpoint or "ema_updates" not in checkpoint:
            raise ValueError("v802 恢复必须包含原 EMA 状态，不能从原始权重重新初始化")
        ema.restore(checkpoint["ema_state_dict"], checkpoint["ema_updates"], state["optimizer_steps"])
        if state_digest(checkpoint["initial_backbone_state_dict"]) != config["initialization"]["state_dict_sha256"]:
            raise ValueError("恢复记录的原始初始化张量改变")
        restore_rng(checkpoint, device)
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
        if ema.updates != state["optimizer_steps"]:
            raise ValueError("保存前 EMA 与优化步数不一致")
        atomic_torch_save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "ema_state_dict": ema.model.state_dict(), "ema_updates": ema.updates,
                           "initial_backbone_state_dict": initial_backbone,
                           "config": config, "progress": state, "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                           "catalog_sha256": catalog_digest}, run / "last.pt")
        write_json(run / "progress.json", {k: state[k] for k in (
            "epoch", "month_cursor", "sample_cursor", "section_cursor", "optimizer_steps", "best_epoch", "stale")})

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
                order = shuffled_sample_indices(block, targets, config["seed"], epoch, month)
                for offset, indices, windows, clock, target in training_batches(
                        block, order, targets, state["section_cursor"], spec["batch_size"], device):
                    step = training_step(model, optimizer, windows, clock, target, config, amp)
                    ema.update(epoch)
                    state["loss_sum"] += step["loss"] * step["rows"]
                    state["main_loss_sum"] += step["main_loss"] * step["rows"]
                    state["ic_loss_sum"] += step["ic_loss"] * step["rows"]
                    state["rows"] += step["rows"]
                    state["optimizer_steps"] += 1
                    state["epoch_steps"] += 1
                    month_batches += 1
                    state["section_cursor"] = offset + 1
                    state["sample_cursor"] += len(indices)
                    if state["epoch_steps"] % spec["checkpoint_batches"] == 0:
                        save_progress()
                for record in skipped.values():
                    reason = record["reason"]
                    state["skipped"][reason] = state["skipped"].get(reason, 0) + 1
                state["sections"] += catalog["months"][month]["sections"] - len(skipped)
                state.update(month_cursor=state["month_cursor"] + 1, sample_cursor=0, section_cursor=0)
                save_progress()
                log_event(run, "trained_month", epoch=epoch, month=month, month_cursor=state["month_cursor"],
                          complete_cross_sections=len(order), batches_updated=month_batches,
                          optimizer_steps=state["optimizer_steps"],
                          seconds=round(time.perf_counter() - month_start, 3))
                del block
        validation_started = time.perf_counter()
        predictions, keys, result = predict_split(store, "validation", ema.model, device, config, run)
        metric = result[0]["ic_mean"]
        validation_loss, model_undefined = huber_loss_check(predictions, keys, store.protocol, config)
        validation_seconds = time.perf_counter() - validation_started
        if metric is None:
            raise ValueError("验证 IC 无定义，不能选择权重")
        eligible = model_undefined == 0 and result[0]["coverage_violations"] == 0
        improved = select_epoch(state, metric, epoch, eligible=eligible)
        if improved:
            atomic_torch_save({"state_dict": {k: v.cpu() for k, v in ema.model.state_dict().items()},
                               "config": config, "protocol": store.protocol, "epoch": epoch,
                               "weight_source": "ema", "ema_updates": ema.updates,
                               "initial_backbone_state_dict": initial_backbone}, run / "best.pt")
            save_evaluation(run, "validation", predictions, result)
        if state["rows"] == 0:
            raise ValueError("整轮没有可优化的样本")
        train_loss = state["loss_sum"] / state["rows"]
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
               "train_huber_loss": state["main_loss_sum"] / state["rows"],
               "train_ic_loss": state["ic_loss_sum"] / state["rows"],
               "training_batching": spec["batching"], "batch_size": spec["batch_size"],
               "train_loss_reduction": spec["loss_reduction"],
               "validation_huber_loss": validation_loss, "validation_ic_mean": metric,
               "validation_weights": "ema", "ema_updates": ema.updates,
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
        state["history"].append(row)
        pd.DataFrame(state["history"]).to_csv(run / "history.csv", index=False, encoding="utf-8-sig")
        log_event(run, "epoch_finished", **row)
        record_resources(run, device)
        state.update(epoch=epoch + 1, month_cursor=0, sample_cursor=0, section_cursor=0, loss_sum=0.0, main_loss_sum=0.0, ic_loss_sum=0.0,
                     sections=0, rows=0, epoch_steps=0, skipped={}, epoch_seconds=0.0)
        checkpoint_clock = time.perf_counter()
        save_progress()
    if not (run / "best.pt").exists():
        raise ValueError("没有通过完整验证检查的 checkpoint")
    selection = {"best_epoch": state["best_epoch"], "best_validation_metric": state["best"],
                 "evaluation_weights": "ema", "ema_decay": averaging["decay"],
                 "ema_warmup_epochs": averaging["warmup_epochs"],
                 "best_validation_ic": state["best"], "epochs_completed": len(state["history"]),
                 "selection_metric": "ic_mean", "patience_epochs": patience_epochs(state["best_epoch"]),
                 "early_stopping": dict(spec["early_stopping"]), "stale_epochs": state["stale"],
                 "model_sha256": sha256(run / "best.pt"),
                 "selected_at": now(), "optimizer_steps": state["optimizer_steps"],
                 "training_seconds_this_session": round(time.perf_counter() - started, 3),
                 "stop_reason": "early_stopping" if should_stop(state) else "max_epochs"}
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
        "loss": spec["loss"], "delta": spec["huber_delta"],
        "ic_auxiliary_weight": spec["ic_auxiliary_weight"],
        "evaluation_loss": "main_zscore_huber_only", "backbone_weights": "jointly_finetuned_local_parent",
        "training_weights": "raw", "evaluation_weights": "ema",
        "label_ddof": 0, "prediction_normalized": False, "ic_loss_identity_applicable": False,
        "training_batching": spec["batching"], "batch_size": spec["batch_size"],
        "training_loss_reduction": spec["loss_reduction"], "evaluation_loss_reduction": "equal_cross_sections",
        "selection_metric": "ic_mean", "factor_scale": config["inference"]["factor_scale"],
        "best_validation_huber_loss": next(r["validation_huber_loss"] for r in state["history"]
                                           if r["epoch"] == state["best_epoch"]),
        "test_huber_loss": test_loss,
        "test_model_undefined_sections": undefined, "parent_preserved": bool(preservation),
    })
    manifest.update(status="complete", stage="complete", completed_at=now(), selection=selection,
                    metrics={s: read_json(run / "evaluation" / s / "metrics.json")
                             for s in ("validation", "test")})
    write_json(run / "manifest.json", manifest)
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
    parser.add_argument("--backbone-artifact", type=Path, help="独立复现时的本地父模型文本权重；按冻结张量指纹核对")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-cache-mode", choices=("auto", "resident", "stream"), default="auto",
                        help="评估缓存按预算自动选择常驻或分块；不改变模型和训练配置")
    parser.add_argument("--reproduction-only", action="store_true",
                        help="仅供独立复现工作区使用，显式记录缺少历史比较产物")
    parser.add_argument("--training-confirmed", action="store_true",
                        help="仅在用户明确确认开始训练后使用")
    args = parser.parse_args()
    if not args.training_confirmed:
        parser.error("v802 训练尚待用户确认；确认后才可传入 --training-confirmed")
    config, protocol, run = read_json(args.config), read_json(args.protocol), args.run_dir.resolve()
    validate_training(config, protocol)
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
        manifest = {"run_id": run.name, "solution_id": "v802", "status": "running", "created_at": now(),
                    "parent_solution_id": config["parent_solution_id"],
                    "parent_run_id": config["parent_run_id"], "references": config["references"],
                    "source": snapshot_sources(ROOT, run, solution_id="v802")}
        write_json(run / "manifest.json", manifest)
    store = None
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
        manifest.update(stage="resident_cache")
        write_json(run / "manifest.json", manifest)
        resident = open_resident_store(store, run, prepare_training_targets, args.resident_cache,
                                        lambda event, **fields: log_event(run, event, **fields))
        store = SectionStore(resident, args.backbone_artifact)
        store.evaluation_cache_mode = args.evaluation_cache_mode
        train(store, run, config, manifest, args.resume)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), stopped_at=now())
        write_json(run / "manifest.json", manifest)
        raise
    finally:
        if isinstance(store, SectionStore):
            store.close()


if __name__ == "__main__":
    main()
