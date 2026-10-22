"""训练期已有标签在不同预测时点之间的截面关系；不重算收益或训练模型。"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import now, read_json, sha256, write_json

HOPS = (1, 2, 3)
MINUTES = (570, 600, 630, 660, 690, 810, 840, 870)


def grouped_correlation(x, y, groups, count, minimum):
    valid = np.isfinite(x) & np.isfinite(y)
    g, x, y = groups[valid], x[valid], y[valid]
    n = np.bincount(g, minlength=count).astype(np.float64)
    sx = np.bincount(g, weights=x, minlength=count)
    sy = np.bincount(g, weights=y, minlength=count)
    vx = np.maximum(np.bincount(g, weights=x * x, minlength=count) - sx * sx / np.maximum(n, 1), 0)
    vy = np.maximum(np.bincount(g, weights=y * y, minlength=count) - sy * sy / np.maximum(n, 1), 0)
    covariance = np.bincount(g, weights=x * y, minlength=count) - sx * sy / np.maximum(n, 1)
    denominator = np.sqrt(vx * vy)
    good = (n >= minimum) & (denominator > 0)
    result = np.divide(covariance, denominator, out=np.full(count, np.nan), where=good)
    # 用三个实际日期的直接相关系数核对分组汇总公式。
    available = np.flatnonzero(good)
    for day in available[np.unique(np.linspace(0, len(available) - 1, min(3, len(available)), dtype=int))]:
        selected = g == day
        if not np.isclose(result[day], np.corrcoef(x[selected], y[selected])[0, 1], rtol=0, atol=1e-12):
            raise ValueError("分组相关性与直接计算不一致")
    return result, n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/label-transition-diagnostics-20261022.json")
    args = parser.parse_args()
    run = ROOT / "outputs/runs/v1102-structured-temporal-experts-s42"
    record, protocol = read_json(run / "resident-cache.json"), read_json(run / "protocol.json")
    cache = Path(record["path"])
    names = ("days", "sample_day", "sample_stock", "sample_clock", "targets")
    arrays = {name: np.load(cache / (name + ".npy"), mmap_mode="r", allow_pickle=False) for name in names}
    try:
        plan = read_json(run / "stock-day-training.json")["segments"]["global"]
        if hashlib.sha256(arrays["targets"]).hexdigest() != plan["primary_targets_sha256"]:
            raise ValueError("提供标签的冻结训练变换已改变")
        days = pd.DatetimeIndex(np.asarray(arrays["days"]).astype("datetime64[ns]"))
        years = days.year.to_numpy(dtype=np.int64)
        interval = protocol["splits"]["train"]
        if days.hasnans or not ((days >= interval["start"]) & (days < interval["stop"])).all() or set(years) != {2020, 2021, 2022, 2023}:
            raise ValueError("训练日期范围或纳秒时间解释异常")
        width = int(arrays["sample_stock"].max()) + 1
        identifiers, inverse = np.unique(arrays["sample_day"].astype(np.int64) * width + arrays["sample_stock"], return_inverse=True)
        samples = np.full((len(identifiers), 8), -1, np.int64)
        samples[inverse, arrays["sample_clock"]] = np.arange(len(inverse))
        if hashlib.sha256(samples).hexdigest() != plan["membership_sha256"]:
            raise ValueError("股票日和原预测时点映射不一致")
        labels = np.asarray(arrays["targets"])[samples.clip(0)]
        labels[samples < 0] = np.nan
        groups = identifiers // width
        results = []
        for hop in HOPS:
            for a in range(8 - hop):
                b = a + hop
                corr, n = grouped_correlation(labels[:, a], labels[:, b], groups, len(days), protocol["min_cross_section_size"])
                row = {"hop": hop, "from_clock": f"{MINUTES[a] // 60:02d}:{MINUTES[a] % 60:02d}",
                       "to_clock": f"{MINUTES[b] // 60:02d}:{MINUTES[b] % 60:02d}", "elapsed_minutes": MINUTES[b] - MINUTES[a],
                       "finite_stock_day_pairs": int(n.sum()), "valid_dates": int(np.isfinite(corr).sum()),
                       "mean_cross_section_correlation": float(np.nanmean(corr)),
                       "negative_date_fraction": float(np.mean(corr[np.isfinite(corr)] < 0)),
                       "annual": {str(year): {"dates": int(np.isfinite(corr[years == year]).sum()),
                                             "mean_cross_section_correlation": float(np.nanmean(corr[years == year]))}
                                  for year in sorted(set(years))}}
                results.append(row)
                if hop == 1:
                    print({"from": row["from_clock"], "to": row["to_clock"],
                           "correlation": row["mean_cross_section_correlation"],
                           "annual": {year: values["mean_cross_section_correlation"] for year, values in row["annual"].items()}}, flush=True)
        result = {"computed_at": now(), "script_sha256": sha256(__file__), "source_run": run.name,
                  "scope": "仅2020—2023训练期；对同日两个预测时点，在共同有效股票上计算标签相关性，再对日期等权平均。",
                  "label_source": "已提供vwap_return经原训练截面标准化后的冻结目标；未从价格重算收益。逐截面Pearson对这种正仿射变换不变。",
                  "hops_fixed_before_analysis": list(HOPS), "training_interval": interval,
                  "training_days": len(days), "stock_days": len(identifiers), "sample_rows": len(inverse),
                  "finite_targets": int(np.isfinite(labels).sum()), "data_manifest_sha256": sha256(run / "data-manifest.json"),
                  "primary_targets_sha256": plan["primary_targets_sha256"], "membership_sha256": plan["membership_sha256"],
                  "results": results, "exposure_read": False, "price_data_read": False,
                  "formal_metrics_modified": False, "formal_training_performed": False, "new_candidate_created": False,
                  "limitations": ["这是标签之间的事后统计关系，不是只用过去行情取得的预测能力。",
                                  "同日预测位置的间隔跨午休时超过30分钟，不把位移步数当成统一分钟数。",
                                  "未根据该结果改写、合成或替换正式评估标签。",
                                  "相邻标签有关系，不证明教师信号可迁移到因果学生；单变量相关性也不排除非线性关系。"]}
        write_json(args.output, result)
    finally:
        for array in arrays.values():
            array._mmap.close()


if __name__ == "__main__":
    main()
