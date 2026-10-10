"""v1 完整训练入口。先准备训练/验证数据，选定权重后才读取测试集。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

# 在仓库或平铺的平台提交包内均能导入相同实现。
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] if (HERE.parents[1] / "docs").is_dir() else HERE
for directory in [PROJECT, PROJECT / "common", PROJECT / "evaluation"]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# Windows 的 CUDA 构建需在 NumPy/OpenMP 初始化前加载其 DLL。
# 只准备数据时仍允许在没有 PyTorch 的本地查询环境中执行。
if "--prepare-only" not in sys.argv:
    import torch

import numpy as np
import pandas as pd

from artifacts import environment, log_event, now, read_json, sha256, snapshot_sources, write_json
from baseline_evaluation import evaluate
from v1_data import KEYS, load_month, month_intervals
from v1_model import NumpyMLP, create_model, export_model


def prepare_split(run, split, config, protocol, datasources, dai_module):
    directory = run / "cache" / split
    directory.mkdir(parents=True, exist_ok=True)
    interval = protocol["splits"][split]
    entries = []
    for lo, hi in month_intervals(interval["start"], interval["stop"]):
        month = lo.strftime("%Y-%m")
        x_path, keys_path = directory / f"{month}.npy", directory / f"{month}.parquet"
        record_path = directory / f"{month}.json"
        if record_path.exists():
            record = read_json(record_path)
            if sha256(x_path) != record["x_sha256"] or sha256(keys_path) != record["keys_sha256"]:
                raise ValueError(f"缓存内容已改变: {split}/{month}")
        else:
            begin = time.perf_counter()
            x, keys, quality = load_month(dai_module, datasources, lo, hi, config, protocol)
            np.save(x_path, x, allow_pickle=False)
            keys.to_parquet(keys_path, index=False)
            record = {"month": month, **quality, "finite_labels": int(np.isfinite(keys[protocol["label_column"]]).sum()),
                      "x_sha256": sha256(x_path), "keys_sha256": sha256(keys_path)}
            write_json(record_path, record)
            log_event(run, "prepared_month", split=split, month=month, seconds=round(time.perf_counter() - begin, 3), **quality)
        entries.append(record)
    write_json(directory / "index.json", entries)
    return entries


def load_split(run, split):
    directory = run / "cache" / split
    entries = read_json(directory / "index.json")
    features = np.concatenate([np.load(directory / f"{row['month']}.npy", allow_pickle=False) for row in entries])
    keys = pd.concat([pd.read_parquet(directory / f"{row['month']}.parquet") for row in entries], ignore_index=True)
    return features, keys


def predict_torch(model, x, device, scale, batch=32768):
    import torch

    result = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            values = x[start:start + batch]
            if not torch.is_tensor(values):
                values = torch.from_numpy(values)
            prediction = model(values.to(device)).squeeze(1) / scale
            result[start:start + len(values)] = prediction.cpu().numpy()
    return result


def save_evaluation(run, split, x, keys, model, protocol):
    directory = run / "evaluation" / split
    directory.mkdir(parents=True, exist_ok=True)
    predictions = keys[KEYS].copy()
    predictions["factor"] = model.predict(x)
    stats, sections, monthly = evaluate(predictions, keys, protocol)
    write_json(directory / "metrics.json", stats)
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    sections.to_parquet(directory / "cross-sections.parquet", index=False)
    monthly.to_csv(directory / "monthly.csv", index=False, encoding="utf-8-sig")
    log_event(run, "evaluated", split=split, **stats)
    return stats


def draw_history(run, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train Huber")
    axes[0].plot(frame["epoch"], frame["validation_loss"], label="Validation Huber")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["rank_ic_mean"], label="Validation Rank IC")
    axes[1].plot(frame["epoch"], frame["ic_mean"], label="Validation IC")
    axes[1].set(xlabel="Epoch", ylabel="Cross-sectional mean")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(run / "learning-curves.png", dpi=160)
    plt.close(fig)


def train(run, config, protocol, datasources, dai_module):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as functional

    seed, spec = config["seed"], config["training"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = environment()
    env.update(device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
               device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
               deterministic_algorithms=True, cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    write_json(run / "environment.json", env)
    (run / "requirements-lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in env["packages"].items()) + "\n", encoding="utf-8")
    x_train, keys_train = load_split(run, "train")
    x_validation, keys_validation = load_split(run, "validation")
    labels = keys_train[protocol["label_column"]].to_numpy(dtype=np.float32)
    valid = np.isfinite(labels)
    training_x = torch.from_numpy(x_train[valid]).to(device)
    training_y = torch.from_numpy(np.clip(labels[valid], -spec["target_clip"], spec["target_clip"]) * spec["target_scale"]).to(device)
    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    log_event(run, "training_started", device=str(device), train_rows=len(training_y), validation_rows=len(keys_validation),
              parameters=sum(p.numel() for p in model.parameters()), seed=seed)
    best, best_epoch, stale, history = -float("inf"), 0, 0, []
    batch = spec["batch_size"]
    started = time.perf_counter()
    for epoch in range(1, spec["epochs"] + 1):
        begin = time.perf_counter()
        model.train()
        order = torch.randperm(len(training_y), device=device)
        total_loss = torch.zeros((), device=device)
        for offset in range(0, len(order), batch):
            index = order[offset:offset + batch]
            optimizer.zero_grad(set_to_none=True)
            output = model(training_x[index]).squeeze(1)
            loss = functional.huber_loss(output, training_y[index], delta=spec["huber_delta"])
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失为非有限值")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.detach() * len(index)
        predictions = keys_validation[KEYS].copy()
        predictions["factor"] = predict_torch(model, x_validation, device, spec["target_scale"])
        stats, _, _ = evaluate(predictions, keys_validation, protocol)
        actual = keys_validation[protocol["label_column"]].to_numpy(dtype=np.float32)
        selected = np.isfinite(actual)
        val_prediction = torch.from_numpy(predictions["factor"].to_numpy(dtype=np.float32)[selected] * spec["target_scale"])
        val_target = torch.from_numpy(np.clip(actual[selected], -spec["target_clip"], spec["target_clip"]) * spec["target_scale"])
        val_loss = functional.huber_loss(val_prediction, val_target, delta=spec["huber_delta"]).item()
        metric = stats[protocol["selection_metric"]]
        if metric is None or not np.isfinite(metric):
            raise ValueError("验证指标无有效值")
        improved = metric > best + spec["min_delta"]
        if improved:
            best, best_epoch, stale = metric, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch}, run / "best.pt")
            export_model(model, config, run / "v1-model.json")
        else:
            stale += 1
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "config": config, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, run / "last.pt")
        row = {"epoch": epoch, "train_loss": float(total_loss.item() / len(training_y)), "validation_loss": val_loss,
               "ic_mean": stats["ic_mean"], "rank_ic_mean": metric, "best_epoch": best_epoch,
               "improved": improved, "seconds": round(time.perf_counter() - begin, 3)}
        history.append(row)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False, encoding="utf-8-sig")
        log_event(run, "epoch_finished", **row)
        if stale >= spec["patience"]:
            log_event(run, "early_stopping", epoch=epoch, best_epoch=best_epoch)
            break
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    exported = NumpyMLP(run / "v1-model.json")
    probe = x_validation[: min(32768, len(x_validation))]
    torch_values = predict_torch(model, probe, device, spec["target_scale"])
    numpy_values = exported.predict(probe)
    np.testing.assert_allclose(numpy_values, torch_values, atol=1e-7, rtol=1e-4)
    parity = {"rows": len(probe), "max_abs_difference": float(np.abs(numpy_values - torch_values).max()), "passed": True}
    write_json(run / "inference-parity.json", parity)
    selection = {"best_epoch": best_epoch, "best_validation_rank_ic": best, "epochs_completed": len(history),
                 "selection_metric": protocol["selection_metric"], "model_sha256": sha256(run / "v1-model.json"),
                 "selected_at": now(), "training_seconds": round(time.perf_counter() - started, 3)}
    write_json(run / "selection.json", selection)
    log_event(run, "model_frozen", **selection)
    draw_history(run, history)
    del training_x, training_y, model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summaries = {}
    summaries["train"] = save_evaluation(run, "train", x_train, keys_train, exported, protocol)
    del x_train, keys_train
    gc.collect()
    summaries["validation"] = save_evaluation(run, "validation", x_validation, keys_validation, exported, protocol)
    del x_validation, keys_validation
    gc.collect()
    # 测试数据第一次进入本次运行发生在权重冻结后。
    prepare_split(run, "test", config, protocol, datasources, dai_module)
    x_test, keys_test = load_split(run, "test")
    summaries["test"] = save_evaluation(run, "test", x_test, keys_test, exported, protocol)
    write_json(run / "metrics.json", summaries)
    manifest = read_json(run / "manifest.json")
    manifest.update(status="complete", completed_at=now(), selection=selection,
                    artifacts={name: sha256(run / name) for name in ["v1-model.json", "best.pt", "last.pt", "history.csv", "metrics.json"]})
    write_json(run / "manifest.json", manifest)
    log_event(run, "run_completed", run_id=run.name)


def initialize(run, config_path, protocol_path, datasources):
    if (run / "manifest.json").exists():
        manifest = read_json(run / "manifest.json")
        if manifest["status"] not in {"prepared", "preparing"}:
            raise ValueError("运行已开始训练或完成，请使用新的运行目录")
        if read_json(config_path) != read_json(run / "config.json") or read_json(protocol_path) != read_json(run / "protocol.json"):
            raise ValueError("已有运行的配置或协议不能改变")
        if datasources != read_json(run / "datasources.json"):
            raise ValueError("已有运行的数据源不能改变")
        for relative, digest in manifest["source"]["files"].items():
            if sha256(PROJECT / relative) != digest:
                raise ValueError(f"准备数据后源码已改变，请恢复快照或新建运行: {relative}")
        if manifest["source"]["local_dai_sha256"] is not None and sha256(PROJECT / "dai.py") != manifest["source"]["local_dai_sha256"]:
            raise ValueError("本地 DAI 实现已改变")
        return
    run.mkdir(parents=True, exist_ok=False)
    config, protocol = read_json(config_path), read_json(protocol_path)
    intervals = [(pd.Timestamp(row["start"]), pd.Timestamp(row["stop"])) for row in protocol["splits"].values()]
    if any(a >= b for a, b in intervals) or any(intervals[i][1] > intervals[i + 1][0] for i in range(len(intervals) - 1)):
        raise ValueError("数据划分必须严格有序且不重叠")
    write_json(run / "config.json", config)
    write_json(run / "protocol.json", protocol)
    write_json(run / "datasources.json", datasources)
    source = snapshot_sources(PROJECT, run)
    manifest = {"run_id": run.name, "solution_id": config["solution_id"], "created_at": now(), "status": "preparing",
                "source": source, "config_sha256": sha256(run / "config.json"), "protocol_sha256": sha256(run / "protocol.json")}
    write_json(run / "manifest.json", manifest)


def record_data(run, datasources, protocol, dai_module):
    if not getattr(dai_module, "IS_LOCAL_SHIM", False):
        write_json(run / "data-manifest.json", {"mode": "platform", "datasources": datasources,
                   "limitation": "平台表不能以本地文件哈希标识；需另存平台数据快照标识"})
        return
    root = Path(os.environ.get("JYCAPITAL_DATA_DIR", str(PROJECT / "data")))
    if not root.is_absolute():
        root = PROJECT / root
    paths = []
    for table in [*datasources.values(), protocol["universe_table"], protocol["label_table"]]:
        target = root / table
        paths.extend(sorted(target.glob("*.parquet")) if target.is_dir() else [target.with_suffix(".parquet")])
    path = run / "data-manifest.json"
    previous = read_json(path) if path.exists() else None
    records = []
    for item in paths:
        record = {"path": item.relative_to(root).as_posix(), "bytes": item.stat().st_size, "sha256": sha256(item)}
        records.append(record)
    current = {"mode": "local", "files": records}
    if previous is not None and current != previous:
        raise ValueError("本次运行使用的数据文件内容已改变")
    write_json(path, current)
    log_event(run, "data_fingerprinted", files=len(records), bytes=sum(row["bytes"] for row in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    default_protocol = PROJECT / "evaluation" / "protocols" / "local-v1.json"
    parser.add_argument("--protocol", default=str(default_protocol if default_protocol.exists() else HERE / "protocol.json"))
    parser.add_argument("--bar-table", default="cpt_jyc_2026_stock_bar5m")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    import dai
    run = Path(args.run_dir).resolve()
    config, protocol = read_json(args.config), read_json(args.protocol)
    datasources = {config["data"]["logical_source"]: args.bar_table}
    initialize(run, args.config, args.protocol, datasources)
    record_data(run, datasources, protocol, dai)
    for split in ["train", "validation"]:
        prepare_split(run, split, config, protocol, datasources, dai)
    manifest = read_json(run / "manifest.json")
    manifest["status"] = "prepared" if args.prepare_only else "training"
    write_json(run / "manifest.json", manifest)
    if not args.prepare_only:
        train(run, config, protocol, datasources, dai)


if __name__ == "__main__":
    main()
