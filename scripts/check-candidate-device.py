"""在训练间隙检查候选默认批量的显存、梯度与推理；只使用合成输入。"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from research_queue import atomic_json, safe_path, source_files, validate_spec
from artifacts import now, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    directory = safe_path(ROOT, "solutions/" + args.solution_id)
    spec = validate_spec(ROOT, directory / "experiment.json")
    if args.device == "cuda":
        import psutil
        for process in psutil.process_iter(["pid", "cmdline"]):
            command = process.info["cmdline"] or []
            if "--training-confirmed" in command and any(Path(value).name.endswith("_train.py") for value in command):
                raise RuntimeError(f"正式训练进程 {process.pid} 仍在运行，GPU 预检须等待模型间隙")
    import torch
    sys.path.insert(0, str(directory))
    trainer = importlib.import_module(args.solution_id + "_train")
    model_module = importlib.import_module(args.solution_id + "_model")
    config = read_json(directory / "model-config.json")
    trainer.validate_training(config, read_json(ROOT / "evaluation/protocols/local-v2-ic.json"))
    class_names = {"patch_transformer": "MultiFrequencyPatchTransformer", "factorized_mixer": "MultiFrequencyMixer",
                   "hierarchical_gru": "MultiFrequencyGRU", "snapshot_mlp": "MultiFrequencySnapshotMLP"}
    torch.set_num_threads(config["training"]["cpu_threads"] if args.device == "cuda" else 1)
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(config["training"]["deterministic"])
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        torch.cuda.reset_peak_memory_stats()
    model = getattr(model_module, class_names[config["model"]["family"]])(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    grouped = config["training"]["batching"].endswith("_stock_days")

    def inputs(count):
        if hasattr(trainer, "preflight_inputs"):
            return trainer.preflight_inputs(config, count, device)
        windows = {f: torch.randn(count, config["model"]["input_channels"], length, device=device)
                   for f, length in config["data"]["window_bars"].items()}
        clock_ids = torch.full((count,), 7, device=device) if grouped else torch.arange(count, device=device) % 8
        clock = torch.nn.functional.one_hot(clock_ids, 8).float()
        for value in windows.values():
            value[:, 72:75] = 1
        return windows, clock

    train_batch = config["training"]["batch_size"] if device.type == "cuda" else 8
    inference_batch = config["inference"]["batch_size"] if device.type == "cuda" else 16
    if grouped:
        train_batch = config["training"]["stock_day_batch_size"] if device.type == "cuda" else 2
        inference_batch = config["inference"]["stock_day_batch_size"] if device.type == "cuda" else 4
    windows, clock = inputs(train_batch)
    targets = torch.linspace(-2, 2, train_batch, dtype=torch.float64, device=device)
    if grouped:
        targets = torch.linspace(-2, 2, train_batch * 16, dtype=torch.float64, device=device).reshape(train_batch, 8, 2)
    if hasattr(trainer, "preflight_targets"):
        targets = trainer.preflight_targets(config, train_batch, device)
    elapsed = []
    for _ in range(3):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer.training_step(model, optimizer, windows, clock, targets, config,
                                       device.type == "cuda" and torch.cuda.is_bf16_supported())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError("参数梯度缺失或非有限")
    del windows, clock, targets
    model.eval()
    with torch.inference_mode():
        predictions = model.forward_group(*inputs(inference_batch))[..., 0] if grouped else model(*inputs(inference_batch))
    if predictions.dtype != torch.float32 or not torch.isfinite(predictions).all():
        raise FloatingPointError("FP32 推理检查失败")
    if hasattr(trainer, "preflight_encoder"):
        trainer.preflight_encoder(model, config, device)
    record = {"solution_id": args.solution_id, "checked_at": now(), "status": "passed", "scope": "synthetic_device_preflight",
              "device": str(device), "train_batch": train_batch, "inference_batch": inference_batch,
              "batch_unit": ("stock_day" if grouped else "stocks_in_complete_cross_section"
                             if config["training"]["batching"] == "global_shuffled_cross_sections" else "sample"),
              "train_target_slots": train_batch * (8 if grouped else 1),
              "inference_output_slots": int(predictions.numel()),
              "parameters": sum(p.numel() for p in model.parameters()), "step_seconds": elapsed,
              "source_files": source_files(ROOT, args.solution_id),
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "formal_training_performed": False}
    path = ROOT / "outputs/checks/continuous-research" / (args.solution_id + "-" + args.device + "-preflight.json")
    atomic_json(path, record)
    print(json.dumps({k: v for k, v in record.items() if k != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
