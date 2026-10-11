"""v2 单次正式训练、验证选择、冻结后测试。行情缓存不进入提交包。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Windows CUDA DLL 必须先于 NumPy/OpenMP 加载。
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT / "common", ROOT / "evaluation"):
    sys.path.insert(0, str(directory))
from artifacts import environment, log_event, now, read_json, sha256, snapshot_sources, write_json
from baseline_evaluation import evaluate
from v2_data import ProcessedStore, KEYS, FREQUENCIES
from v2_model import MultiFrequencyTCN


def atomic_torch_save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


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
                values[index] = (model(windows, clock) / config["training"]["target_scale"]).cpu().numpy()
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
        if read_json(source / "protocol.json") != read_json(run / "protocol.json"):
            raise ValueError("预处理与模型评估协议不同")
        references[name] = {"path": str(source.resolve()), "manifest_sha256": sha256(source / "manifest.json")}
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


def train(store, run, config, manifest, resume=False):
    device = configure(config)
    spec = config["training"]
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = MultiFrequencyTCN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    env = environment()
    env.update(device=str(device), device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
               torch=torch.__version__, cuda=torch.version.cuda, training_autocast_bfloat16=amp,
               evaluation_precision="float32", parameters=sum(p.numel() for p in model.parameters()))
    write_json(run / "environment.json", env)
    (run / "requirements-lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in env["packages"].items()) + "\n", encoding="utf-8")
    state = {"epoch": 1, "month_cursor": 0, "best": -float("inf"), "best_epoch": 0,
             "patience_best": -float("inf"), "stale": 0, "history": [], "loss_sum": 0.0, "rows": 0}
    if resume and (run / "last.pt").exists():
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("续跑配置不同")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["progress"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    log_event(run, "training_started", parameters=env["parameters"], device=env["device_name"], resume=resume)
    started = time.perf_counter()

    def save_progress():
        atomic_torch_save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                           "progress": state, "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, run / "last.pt")

    for epoch in range(state["epoch"], spec["epochs"] + 1):
        if state["stale"] >= spec["patience"]:
            break
        begin = time.perf_counter()
        months = np.random.default_rng(config["seed"] + epoch).permutation(store.months("train")).tolist()
        remaining = months[state["month_cursor"]:]
        if remaining:
            for month, block in blocks(store, "train", remaining):
                month_start = time.perf_counter()
                model.train()
                actual = block.keys[store.protocol["label_column"]].to_numpy(dtype=np.float32)
                index = np.flatnonzero(np.isfinite(actual))
                generator = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, int(month.replace("-", ""))]))
                generator.shuffle(index)
                targets = np.clip(actual, -spec["target_clip"], spec["target_clip"]) * spec["target_scale"]
                losses = torch.zeros((), device=device)
                for offset in range(0, len(index), spec["batch_size"]):
                    selected = index[offset:offset + spec["batch_size"]]
                    windows, clock = tensor_batch(block, selected, device)
                    y = torch.from_numpy(targets[selected]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                        prediction = model(windows, clock)
                        loss = F.huber_loss(prediction, y, delta=spec["huber_delta"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("非有限训练损失")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    losses += loss.detach() * len(selected)
                state["loss_sum"] += losses.item()
                state["rows"] += len(index)
                state["month_cursor"] += 1
                save_progress()
                log_event(run, "trained_month", epoch=epoch, month=month, month_cursor=state["month_cursor"],
                          rows=len(index), loss=losses.item() / len(index), seconds=round(time.perf_counter() - month_start, 3))
                del block
        predictions, keys, result = predict_split(store, "validation", model, device, config, run)
        metric = result[0][store.protocol["selection_metric"]]
        if metric is None or not np.isfinite(metric):
            raise ValueError("验证主指标无有效值")
        actual = keys[store.protocol["label_column"]].to_numpy(dtype=np.float32)
        take = np.isfinite(actual)
        target = torch.from_numpy(np.clip(actual[take], -spec["target_clip"], spec["target_clip"]) * spec["target_scale"])
        output = torch.from_numpy(predictions.factor.to_numpy(dtype=np.float32)[take] * spec["target_scale"])
        validation_loss = F.huber_loss(output, target, delta=spec["huber_delta"]).item()
        improved = metric > state["best"]
        if improved:
            state.update(best=metric, best_epoch=epoch)
            atomic_torch_save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": config, "epoch": epoch}, run / "best.pt")
            save_evaluation(run, "validation", predictions, result)
        if metric > state["patience_best"] + spec["min_delta"]:
            state.update(patience_best=metric, stale=0)
        else:
            state["stale"] += 1
        row = {"epoch": epoch, "train_loss": state["loss_sum"] / state["rows"], "validation_loss": validation_loss,
               "ic_mean": result[0]["ic_mean"], "rank_ic_mean": metric, "best_epoch": state["best_epoch"],
               "improved": improved, "train_rows": state["rows"], "seconds": round(time.perf_counter() - begin, 3)}
        state["history"].append(row)
        pd.DataFrame(state["history"]).to_csv(run / "history.csv", index=False, encoding="utf-8-sig")
        log_event(run, "epoch_finished", **row)
        state.update(epoch=epoch + 1, month_cursor=0, loss_sum=0.0, rows=0)
        save_progress()
    selection = {"best_epoch": state["best_epoch"], "best_validation_rank_ic": state["best"],
                 "epochs_completed": len(state["history"]), "selection_metric": store.protocol["selection_metric"],
                 "model_sha256": sha256(run / "best.pt"), "selected_at": now(),
                 "training_seconds_this_session": round(time.perf_counter() - started, 3),
                 "stop_reason": "early_stopping" if state["stale"] >= spec["patience"] else "max_epochs"}
    write_json(run / "selection.json", selection)
    log_event(run, "model_frozen", **selection)
    draw_history(run, state["history"])
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    # 到这里才读取测试标签，测试绝不参与权重选择。
    predictions, _, result = predict_split(store, "test", model, device, config, run)
    save_evaluation(run, "test", predictions, result)
    manifest.update(status="complete", completed_at=now(), selection=selection,
                    metrics={s: read_json(run / "evaluation" / s / "metrics.json") for s in ("validation", "test")})
    write_json(run / "manifest.json", manifest)
    log_event(run, "completed", validation_rank_ic=state["best"], test_rank_ic=result[0]["rank_ic_mean"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "model-config.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "evaluation/protocols/local-v1.json")
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--holdout-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config, protocol, run = read_json(args.config), read_json(args.protocol), args.run_dir.resolve()
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
            if path.startswith("solutions/v2/") and sha256(ROOT / path) != digest:
                raise ValueError(f"续跑源码已改变: {path}")
    else:
        write_json(run / "config.json", config)
        write_json(run / "protocol.json", protocol)
        manifest = {"run_id": run.name, "solution_id": "v2", "status": "running", "created_at": now(),
                    "parent_run_id": config["parent_run_id"], "references": {"local_best": config["parent_run_id"], "platform_best": None},
                    "source": snapshot_sources(ROOT, run, solution_id="v2")}
        write_json(run / "manifest.json", manifest)
    try:
        store = ProcessedStore(args.training_run, args.holdout_run, args.data_root, config, protocol)
        data_manifest(store, run)
        train(store, run, config, manifest, args.resume)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), stopped_at=now())
        write_json(run / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
