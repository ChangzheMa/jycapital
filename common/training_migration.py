"""月内采样到全局 FP16 的显式训练迁移；来源运行始终只读。"""

from __future__ import annotations

import copy
import ast
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from artifacts import read_json, sha256, write_json, now

KEYS = ["date", "instrument"]


def validate_transition(old, new, old_protocol, new_protocol):
    if old_protocol != new_protocol:
        raise ValueError("迁移不能改变评估协议")
    if old["solution_id"] != "v101" or new["solution_id"] != "v101":
        raise ValueError("此迁移入口仅适用于 v101")
    if old["seed"] != new["seed"] or old["model"] != new["model"]:
        raise ValueError("迁移不能改变种子或模型结构")
    old_data, new_data = copy.deepcopy(old["data"]), copy.deepcopy(new["data"])
    if old_data.pop("input_precision", "float32") != "float32":
        raise ValueError("迁移来源必须使用 FP32 输入")
    if new_data.pop("input_precision", None) != "fp16_with_fp32_overflow" or old_data != new_data:
        raise ValueError("迁移只能改变输入舍入，不能改变窗口或字段")
    old_training, new_training = copy.deepcopy(old["training"]), copy.deepcopy(new["training"])
    if old_training.pop("batching") != "monthly_shuffled_samples":
        raise ValueError("迁移来源必须为月内采样")
    if new_training.pop("batching") != "global_shuffled_samples":
        raise ValueError("迁移目标必须为全局采样")
    new_training.pop("resident_cache", None)
    old_training.pop("resident_cache", None)
    if old_training != new_training or old["inference"] != new["inference"]:
        raise ValueError("迁移不能改变优化器、损失、batch、早停或累计轮数上限")


def source_record(source):
    source = Path(source).resolve()
    manifest, config, protocol = [read_json(source / name) for name in ("manifest.json", "config.json", "protocol.json")]
    if manifest.get("status") != "interrupted":
        raise ValueError("迁移来源必须为明确中断的运行")
    files = {name: sha256(source / name) for name in (
        "manifest.json", "config.json", "protocol.json", "last.pt", "best.pt", "data-manifest.json", "structural-sections.json")}
    last = torch.load(source / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    if last["config"] != config or best["config"] != config:
        raise ValueError("来源 checkpoint 与冻结配置不一致")
    if last["catalog_sha256"] != files["structural-sections.json"]:
        raise ValueError("来源标签截面清单已改变")
    if best["epoch"] != last["progress"]["best_epoch"]:
        raise ValueError("来源最佳权重轮次与恢复状态不一致")
    if not last.get("optimizer", {}).get("state") or "torch_rng" not in last or "cuda_rng" not in last:
        raise ValueError("来源缺少完整优化器或随机状态")
    for relative, expected in manifest["source"]["files"].items():
        if sha256(source / "source" / relative) != expected:
            raise ValueError("来源冻结源码已改变: " + relative)
    return {"source_run_id": source.name, "source_path": str(source), "source_files": files,
            "source_config": config, "source_protocol": protocol}, last, best


def validate_implementation(source, solution):
    old = Path(source) / "source/solutions/v101"
    solution = Path(solution)
    if sha256(old / "v101_model.py") != sha256(solution / "v101_model.py"):
        raise ValueError("迁移不能改变模型实现")
    for filename, names in {"v101_train.py": ["prepare_training_targets"],
                            "v101_loss.py": ["standardized_targets", "sample_huber_loss"],
                            "v101_data.py": ["section_indices"]}.items():
        definitions = []
        for path in (old / filename, solution / filename):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            definitions.append({n.name: ast.dump(n, include_attributes=False) for n in tree.body
                                if isinstance(n, ast.FunctionDef) and n.name in names})
        if definitions[0] != definitions[1] or len(definitions[0]) != len(names):
            raise ValueError("迁移不能改变标签、损失或截面分组定义: " + filename)


def state_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                visit(element)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def verify_source(record):
    source = Path(record["source_path"])
    for name, expected in record["source_files"].items():
        if sha256(source / name) != expected:
            raise ValueError("迁移来源文件已改变: " + name)


def key_digest(keys, selected):
    """按日期和显式股票字典编码键，避免用位置哈希冒充键哈希。"""
    stocks = pd.Index(sorted(keys.instrument.unique()))
    encoded = np.empty(len(selected), dtype=[("date", "<i8"), ("instrument", "<i4")])
    chosen = keys.iloc[selected]
    encoded["date"] = chosen.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    encoded["instrument"] = stocks.get_indexer(chosen.instrument)
    digest = hashlib.sha256("\n".join(stocks).encode("utf-8"))
    digest.update(encoded.tobytes())
    return digest.hexdigest()


def reconstruct_remaining(keys, targets, config, progress, catalog, protocol):
    """以旧月份排列、旧月内排列和键重建分区，完全不使用新全局排列的前缀。"""
    keys = keys[KEYS].reset_index(drop=True)
    if len(keys) != len(targets) or keys.isna().any().any() or keys.duplicated(KEYS).any():
        raise ValueError("全局样本键长度不符、重复或为空")
    if not keys.equals(keys.sort_values(KEYS).reset_index(drop=True)):
        raise ValueError("全局样本键必须按 date/instrument 规范排序")
    epoch, cursor, sample_cursor = (int(progress[k]) for k in ("epoch", "month_cursor", "sample_cursor"))
    interval = protocol["splits"]["train"]
    months = [str(x) for x in pd.period_range(interval["start"], pd.Timestamp(interval["stop"]) - pd.Timedelta(1, "ns"), freq="M")]
    month_order = np.random.default_rng(config["seed"] + epoch).permutation(months).tolist()
    if not 0 <= cursor <= len(month_order) or sample_cursor < 0 or cursor == len(month_order) and sample_cursor:
        raise ValueError("旧月份或样本游标越界")
    month_codes = keys.date.to_numpy(dtype="datetime64[M]").astype(np.int64)
    eligible = np.isfinite(targets)
    expected_rows = sum(row["rows"] for row in catalog["months"].values())
    expected_eligible = sum(row["finite_label_rows"] - sum(part["finite_labels"] for part in row["skipped"].values())
                            for row in catalog["months"].values())
    if len(keys) != expected_rows or int(eligible.sum()) != expected_eligible:
        raise ValueError("当前样本键或合格标签数量与来源截面清单不同")
    processed = np.zeros(len(keys), dtype=bool)
    steps = 0
    for position, month in enumerate(month_order):
        code = np.datetime64(month, "M").astype(np.int64)
        indices = np.flatnonzero((month_codes == code) & eligible)
        # 原月内排序的相同位置排列，映射到规范全局 (date,instrument) 索引。
        generator = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, int(month.replace("-", ""))]))
        generator.shuffle(indices)
        n = len(indices) if position < cursor else sample_cursor if position == cursor else 0
        if n > len(indices) or n < len(indices) and n % config["training"]["batch_size"]:
            raise ValueError("旧游标不对应完整 batch 边界")
        if position == cursor and n and month not in catalog["months"]:
            raise ValueError("来源截面清单缺少当前月份")
        processed[indices[:n]] = True
        steps += (n + config["training"]["batch_size"] - 1) // config["training"]["batch_size"]
    processed_indices = np.flatnonzero(processed)
    remaining = np.flatnonzero(eligible & ~processed)
    eligible_indices = np.flatnonzero(eligible)
    if len(processed_indices) != progress["rows"] or steps != progress["epoch_steps"]:
        raise ValueError("由旧游标重建的样本数或优化器更新数与 checkpoint 不符")
    if np.any(processed & ~eligible) or len(processed_indices) + len(remaining) != len(eligible_indices):
        raise ValueError("已处理与剩余集合未形成合格标签集合的完整分区")
    if len(progress["history"]) != epoch - 1:
        raise ValueError("累计完整轮次与当前 epoch 不一致")
    audit = {"epoch": epoch, "completed_epochs": len(progress["history"]),
             "legacy_month_order": month_order, "legacy_month_cursor": cursor,
             "legacy_current_month": month_order[cursor] if cursor < len(month_order) else None,
             "legacy_sample_cursor": sample_cursor, "legacy_epoch_optimizer_steps": steps,
             "processed_samples": len(processed_indices), "remaining_samples": len(remaining),
             "eligible_samples": len(eligible_indices), "disjoint": True, "exhaustive": True,
             "processed_keys_sha256": key_digest(keys, processed_indices),
             "remaining_keys_sha256": key_digest(keys, remaining),
             "eligible_keys_sha256": key_digest(keys, eligible_indices),
             "target_values_sha256": hashlib.sha256(np.asarray(targets, dtype=np.float64).tobytes()).hexdigest()}
    return remaining, audit


def remaining_order(run, state, seed, epoch):
    migration = state.get("migration")
    if not migration or epoch != migration["epoch"]:
        return None
    path = Path(run) / "migration-remaining-indices.npy"
    if sha256(path) != migration["remaining_indices_sha256"]:
        raise ValueError("迁移剩余样本集合已改变")
    order = np.load(path, allow_pickle=False).copy()
    np.random.default_rng(np.random.SeedSequence([seed, epoch, 0])).shuffle(order)
    return order


def prepare_state(source, run, config, protocol, catalog, keys, targets, data_manifest):
    record, last, best = source_record(source)
    validate_transition(record["source_config"], config, record["source_protocol"], protocol)
    if read_json(Path(source) / "data-manifest.json") != data_manifest:
        raise ValueError("迁移前后的行情、标签或股票池指纹不同")
    remaining, audit = reconstruct_remaining(keys, targets, record["source_config"], last["progress"],
                                             read_json(Path(source) / "structural-sections.json"), protocol)
    path = Path(run) / "migration-remaining-indices.npy"
    np.save(path, remaining, allow_pickle=False)
    record.update(kind="monthly_fp32_to_global_fp16", prepared_at=now(), sample_handoff=audit,
                  target_config=config, target_input_precision=config["data"]["input_precision"],
                  optimizer_inherited=False, rng_inherited=False,
                  limitation="月内 FP32 训练后迁移到全局 FP16；混合训练轨迹，不是从头全局训练。")
    write_json(Path(run) / "migration.json", record)
    state = copy.deepcopy(last["progress"])
    state["migration"] = {"epoch": audit["epoch"], "source_run_id": record["source_run_id"],
                          "remaining_indices_sha256": sha256(path),
                          "legacy_rows": state["rows"], "legacy_loss_sum": state["loss_sum"],
                          "legacy_seconds": state["epoch_seconds"], "legacy_epoch_steps": state["epoch_steps"],
                          "remaining_samples": len(remaining)}
    # 新阶段游标只指向本轮剩余集合，累计样本、损失和优化器步数保留。
    state.update(month_cursor=0, sample_cursor=0)
    for row in state["history"]:
        row["training_phase"] = "legacy_monthly_fp32"
    return state, record, last, best
