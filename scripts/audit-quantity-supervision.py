"""训练期后续数量状态目标的覆盖与持续性；不读取收益标签、不拟合模型。"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from artifacts import now, read_json, sha256, write_json

CACHE_KEY = "c760f3882029e185c69b39c92509beeeea3995e2214360a6196046d70158563b"
QUANTITY_INDICES = [4, 5, *range(16, 36)]
FIELDS = ["volume", "amount"] + [f"{side}_{kind}{level}" for kind in ("volume", "num_orders")
                                  for side in ("ask", "bid") for level in range(1, 6)]
GROUPS = {"trades": [0, 1], "book_volume": list(range(2, 12)), "book_order_count": list(range(12, 22))}
CLOCKS = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
TARGET_CLOSES = ("10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00")


def decode_block(values, pointers, offsets, exceptions, index):
    result = np.array(values[index], dtype=np.float32, copy=True)
    start, stop = pointers[index:index + 2]
    if stop > start:
        result.reshape(-1)[offsets[start:stop]] = exceptions[start:stop]
    return result


def align_targets(current, previous):
    if current.shape != previous.shape or current.shape[-2:] != (8, 22):
        raise ValueError("数量状态必须为 [...,8,22]")
    latest = np.concatenate([previous[..., -1:, :], current[..., :-1, :]], axis=-2)
    return {"target": current, "last_closed_bar": latest, "previous_day_same_slot": previous}


def correlation(left, right):
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 30:
        return None
    x, y = left[mask].astype(np.float64), right[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt((x @ x) * (y @ y))
    return float((x @ y) / denominator) if denominator > 0 else None


def statistics(target, latest, same_slot):
    """固定日期与预测时点的股票截面；两个基准共享同一有限样本。"""
    valid_target = np.isfinite(target)
    valid = valid_target & np.isfinite(latest) & np.isfinite(same_slot)
    result = {"expected": len(target), "target_finite": int(valid_target.sum()), "matched": int(valid.sum())}
    if valid.sum() < 30:
        return result
    y, last, old = [x[valid].astype(np.float64) for x in (target, latest, same_slot)]
    result.update({"target_square_sum": float(y @ y),
                   "target_centered_square_sum": float(((y - y.mean()) ** 2).sum()),
                   "last_closed_bar_squared_error_sum": float(((last - y) ** 2).sum()),
                   "previous_day_same_slot_squared_error_sum": float(((old - y) ** 2).sum()),
                   "last_closed_bar_correlation": correlation(y, last),
                   "previous_day_same_slot_correlation": correlation(y, old),
                   "innovation_standard_deviation": float((y - last).std()),
                   "target_standard_deviation": float(y.std())})
    return result


def aggregate(rows):
    output = []
    for clock in range(8):
        for group, fields in GROUPS.items():
            selected = [r for r in rows if r["clock_index"] == clock and r["field_index"] in fields]
            usable = [r for r in selected if "target_square_sum" in r]
            expected = sum(r["expected"] for r in selected)
            item = {"clock": CLOCKS[clock], "target_bar_close": TARGET_CLOSES[clock], "group": group,
                    "expected_values": expected, "target_coverage": sum(r["target_finite"] for r in selected) / expected,
                    "matched_baseline_coverage": sum(r["matched"] for r in selected) / expected,
                    "field_sections": len(selected), "usable_field_sections": len(usable)}
            if usable:
                total = sum(r["matched"] for r in usable)
                item["zero_baseline_rmse"] = float(np.sqrt(sum(r["target_square_sum"] for r in usable) / total))
                item["mean_innovation_to_target_std"] = float(np.mean([r["innovation_standard_deviation"] / r["target_standard_deviation"]
                    for r in usable if r["target_standard_deviation"] > 0]))
                for baseline in ("last_closed_bar", "previous_day_same_slot"):
                    values = [r[baseline + "_correlation"] for r in usable if r[baseline + "_correlation"] is not None]
                    item[baseline + "_rmse"] = float(np.sqrt(sum(r[baseline + "_squared_error_sum"] for r in usable) / total))
                    item[baseline + "_mean_quantity_correlation"] = float(np.mean(values)) if values else None
                    item[baseline + "_defined_sections"] = len(values)
            output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/quantity-supervision-audit-20261023.json")
    args = parser.parse_args()
    directory = ROOT / "outputs/preprocessing/resident-cache" / CACHE_KEY
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] != "complete" or manifest["cache_key"] != CACHE_KEY:
        raise ValueError("缓存来源不匹配")
    names = ["days", "sample_day", "sample_clock", "sample_stock", "bar30m-map", "bar30m-values",
             "bar30m-exception-pointers", "bar30m-exception-offsets", "bar30m-exception-values"]
    arrays = {name: np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False) for name in names}
    for name, value in arrays.items():
        record = manifest["files"][name + ".npy"]
        if list(value.shape) != record["shape"] or value.dtype != np.dtype(record["dtype"]):
            raise ValueError("缓存元数据不匹配")
    days = arrays["days"].astype("datetime64[ns]").astype("datetime64[D]").astype(str)
    values, pointer, offset, exception = [arrays["bar30m-" + name] for name in
                                         ("values", "exception-pointers", "exception-offsets", "exception-values")]
    rows, fingerprints = [], []
    for year in range(2020, 2024):
        selected_days = np.flatnonzero(np.char.startswith(days, f"{year}-06-"))[:5]
        first = int(selected_days[0])
        pool = arrays["sample_stock"][(arrays["sample_day"] == first) & (arrays["sample_clock"] == 0)]
        chosen = sorted(np.random.default_rng(np.random.SeedSequence([20261023, year])).choice(pool, 64, replace=False))
        for day in selected_days:
            if not ("2020-01-01" <= days[day] < "2024-01-01") or day < 1:
                raise ValueError("目标必须严格位于正式训练区间且有过去日历槽位")
            current, previous, membership = [], [], []
            day_members = arrays["sample_day"] == day
            day_stocks, day_clocks = arrays["sample_stock"][day_members], arrays["sample_clock"][day_members]
            for stock in chosen:
                ids = arrays["bar30m-map"][[day, day - 1], stock]
                blocks = [decode_block(values, pointer, offset, exception, int(i))[:, QUANTITY_INDICES] for i in ids]
                current.append(blocks[0]); previous.append(blocks[1])
                present = day_clocks[day_stocks == stock]
                membership.append(np.isin(np.arange(8), present))
            current, previous = np.stack(current), np.stack(previous)
            member = np.stack(membership)
            fingerprint = hashlib.sha256(current.tobytes() + previous.tobytes() + member.tobytes()).hexdigest()
            fingerprints.append({"day": days[day], "stock_days": len(chosen), "sample_arrays_sha256": fingerprint,
                                  "stock_indices_sha256": hashlib.sha256(np.asarray(chosen, np.int32).tobytes()).hexdigest()})
            aligned = align_targets(current, previous)
            for clock in range(8):
                for field in range(22):
                    vectors = [value[:, clock, field][member[:, clock]] for value in aligned.values()]
                    rows.append({"year": year, "day": days[day], "clock_index": clock, "field_index": field,
                                 **statistics(*vectors)})
    report = {"computed_at": now(), "script_sha256": sha256(__file__), "cache_key": CACHE_KEY,
        "cache_manifest_sha256": sha256(directory / "manifest.json"),
        "representation": "冻结v1601输入中的训练归一化z数量通道，FP16码及FP32溢出恢复；不是原始数量单位。",
        "fields": FIELDS, "source_channel_indices": QUANTITY_INDICES, "quantity_groups": GROUPS,
        "sampling": "2020—2023各年六月首五个交易日，每年首日09:30股票池随机64只，SeedSequence([20261023,year])；逐预测时点保留当时成员。",
        "stock_days": sum(r["stock_days"] for r in fingerprints), "dates": len(fingerprints),
        "target_alignment": dict(zip(CLOCKS, TARGET_CLOSES)),
        "baselines": "同一股票最后已闭合30分钟Bar，以及前一交易日相同Bar槽位。09:30的前者使用昨日最后Bar；11:30目标为13:30Bar，明确跳过午休，不解释为墙钟30分钟收益。",
        "metric_scope": "各日期、预测时点和数量字段上计算股票截面相关性；汇总为字段截面等权，RMSE在两种基准均有限的共同值上计算。不是vwap_return IC。",
        "model_training_performed": False, "official_return_label_read": False, "target_cache_read": False,
        "validation_or_test_read": False, "exposure_read": False, "cache_modified": False,
        "full_input_cache_rehashed": False, "sample_fingerprints": fingerprints,
        "results": aggregate(rows), "year_results": {str(year): aggregate([r for r in rows if r["year"] == year]) for year in range(2020, 2024)},
        "limitations": ["只测固定训练样本的简单持续性，不是新模型效果或独立样本外验证。",
                        "目标容易由持久性预测并不证明其能够帮助收益预测；辅助模型必须超过相同持续性基准。",
                        "这里只研究已有数量状态，不构造新收益、波动或盘口手工因子，不读取exposure。",
                        "训练期拟合的逐股票标准化、稀疏例外与量化会影响数值；本次没有重算原始行情或训练统计。"]}
    write_json(args.output, report)
    print({"output": str(args.output), "stock_days": report["stock_days"], "field_sections": len(rows)})


if __name__ == "__main__":
    main()
