"""只读股票池索引估算历史30分钟上下文容量，不读取标签或行情数值。"""
import argparse
from pathlib import Path
import sys

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import now, read_json, sha256, write_json


def needed_previous_days(present, horizon):
    """返回每个历史股票日是否被之后 horizon 日内的预测使用，排除预测当日。"""
    present = np.asarray(present, dtype=bool)
    if present.ndim != 2 or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("需要二维股票日矩阵与正整数历史长度")
    prefix = np.vstack([np.zeros((1, present.shape[1]), np.int32), np.cumsum(present, axis=0, dtype=np.int32)])
    days = np.arange(len(present))
    return (prefix[np.minimum(days + 1 + horizon, len(present))] - prefix[days + 1]) > 0


def window_estimate(present, horizon):
    needed = needed_previous_days(present, horizon)
    blocks = int(needed.sum()) + 1
    groups_per_day = np.asarray(present, dtype=bool).sum(axis=1)
    full = int(groups_per_day[np.arange(len(present)) >= horizon].sum())
    total = int(groups_per_day.sum())
    return {
        "previous_trading_days": horizon, "blocks_including_zero": blocks,
        "fp16_values_bytes": blocks * 8 * 72 * 2,
        "fp32_values_bytes": blocks * 8 * 72 * 4,
        "map_bytes": needed.size * 4,
        "training_groups_with_full_calendar_history": full,
        "training_groups_with_shorter_calendar_history": total - full,
        "full_history_group_fraction": full / total if total else None,
        "batch64_fp16_values_bytes": 64 * horizon * 8 * 72 * 2,
        "batch64_fp32_75_channel_bytes": 64 * horizon * 8 * 75 * 4,
        "daily_tokens": horizon, "four_day_patch_tokens": (horizon + 3) // 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/preprocessing/resident-cache/dbcd47ff50672ccee5adc5dec94833d43d5c4226fb540909fa080c71467b960b")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/long-context-feasibility-20261022.json")
    args = parser.parse_args()
    manifest = read_json(args.cache / "manifest.json")
    days = np.load(args.cache / "days.npy", mmap_mode="r", allow_pickle=False)
    sample_day = np.load(args.cache / "sample_day.npy", mmap_mode="r", allow_pickle=False)
    sample_stock = np.load(args.cache / "sample_stock.npy", mmap_mode="r", allow_pickle=False)
    if len(sample_day) != len(sample_stock) or not len(days):
        raise ValueError("缓存索引长度异常")
    if (sample_day < 0).any() or (sample_day >= len(days)).any() or (sample_stock < 0).any() or (sample_stock >= len(manifest["stocks"])).any():
        raise ValueError("缓存索引超出日期或股票范围")
    present = np.zeros((len(days), len(manifest["stocks"])), bool)
    present[sample_day, sample_stock] = True
    result = {
        "checked_at": now(),
        "scope": "只读训练缓存的日期、股票与预测键索引；不读取标签或行情数值，不构建缓存。",
        "input_cache_key": args.cache.name, "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "script_sha256": sha256(__file__), "training_days": len(days), "training_stocks": len(manifest["stocks"]),
        "stock_day_groups": int(present.sum()), "sample_rows": len(sample_day),
        "windows": [window_estimate(present, horizon) for horizon in (20, 60, 120, 252)],
        "original_training_cache_bytes": manifest["resident_array_bytes"],
        "no_label_read": True, "no_exposure_read": True, "new_cache_built": False,
        "limitations": [
            "仅日历和股票池需求估算；不保证每个历史股票日有实际行情，正式输入必须保留缺失掩码。",
            "数值容量未包含极值例外、全部元数据、进程内存及框架开销。",
            "不足完整历史的训练样本保留，不删除或降低覆盖。",
            "只估算训练侧，不包含验证/测试额外缓存，也不证明长上下文提高 IC。",
        ],
    }
    write_json(args.output, result)
    for row in result["windows"]:
        print({"days": row["previous_trading_days"], "fp32_gib": row["fp32_values_bytes"] / 2**30,
               "full_calendar_history_fraction": row["full_history_group_fraction"]})


if __name__ == "__main__":
    main()
