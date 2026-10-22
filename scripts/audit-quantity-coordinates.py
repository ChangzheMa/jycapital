"""只读训练样本，比较盘口数量共同坐标与价格时间路径的表示。"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd
import pyarrow

ROOT = Path(__file__).resolve().parents[1]
for name in ("common", "solutions/v1601"):
    sys.path.insert(0, str(ROOT / name))
from artifacts import now, read_json, sha256, write_json
from v1601_data import minute_slots

KEYS = ["date", "instrument"]
GROUPS = {
    "book_volume": [f"{side}_volume{level}" for side in ("ask", "bid") for level in range(1, 6)],
    "book_order_count": [f"{side}_num_orders{level}" for side in ("ask", "bid") for level in range(1, 6)],
}
FIELDS = [field for group in GROUPS.values() for field in group]
PAIR_TYPES = {
    "same_level_bid_ask": [(level, level + 5) for level in range(5)],
    "same_side_adjacent_levels": [(base + level, base + level + 1) for base in (0, 5) for level in range(4)],
}


def rounded(values):
    with np.errstate(over="ignore", invalid="ignore"):
        half = values.astype(np.float16).astype(np.float32)
    return np.where(np.isfinite(values) & ~np.isfinite(half), values, half)


def pooled_group_parameters(statistics, fields):
    indices = [statistics["fields"].index(field) for field in fields]
    moment = statistics["global"]
    counts = np.asarray(moment["count"], np.float64)[indices]
    means = np.asarray(moment["mean"], np.float64)[indices]
    m2 = np.asarray(moment["m2"], np.float64)[indices]
    total = float(counts.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("训练期全局统计不足")
    center = float((counts * means).sum() / total)
    variance = float((m2 + counts * (means - center) ** 2).sum() / total)
    scale = np.sqrt(variance) if variance > 1e-16 else 1.
    return {"center": center, "scale": float(scale), "observations": int(total)}


def common_quantity_view(raw, parameters):
    valid = np.isfinite(raw) & (raw >= 0)
    safe = np.where(valid, raw, 0)
    return np.where(valid, (np.log1p(safe) - parameters["center"]) / parameters["scale"], np.nan).astype(np.float32)


def pair_counts(raw, view, pairs):
    result = {key: 0 for key in ("valid_pairs", "unequal_raw_pairs", "direction_reversed", "difference_erased",
                                 "raw_ties", "raw_ties_split")}
    for left, right in pairs:
        valid = (np.isfinite(raw[:, left]) & np.isfinite(raw[:, right]) & (raw[:, left] >= 0) & (raw[:, right] >= 0)
                 & np.isfinite(view[:, left]) & np.isfinite(view[:, right]))
        raw_sign = np.sign(raw[valid, right].astype(np.float64) - raw[valid, left].astype(np.float64))
        view_sign = np.sign(view[valid, right].astype(np.float64) - view[valid, left].astype(np.float64))
        unequal = raw_sign != 0
        result["valid_pairs"] += int(valid.sum())
        result["unequal_raw_pairs"] += int(unequal.sum())
        result["direction_reversed"] += int((unequal & (view_sign == -raw_sign)).sum())
        result["difference_erased"] += int((unequal & (view_sign == 0)).sum())
        result["raw_ties"] += int((~unequal).sum())
        result["raw_ties_split"] += int(((~unequal) & (view_sign != 0)).sum())
    return result


def fractions(counts):
    result = dict(counts)
    for key, denominator in (("direction_reversed", "unequal_raw_pairs"), ("difference_erased", "unequal_raw_pairs"),
                             ("raw_ties_split", "raw_ties"), ("z_fp16_erased", "nonzero_close_changes"),
                             ("anchor_fp32_erased", "nonzero_close_changes"),
                             ("global_anchor_fp32_erased", "nonzero_close_changes")):
        if key in counts:
            result[key + "_fraction"] = counts[key] / counts[denominator] if counts[denominator] else None
    return result


def regular(frame, frequency):
    minutes = frame.date.dt.hour * 60 + frame.date.dt.minute
    return frame.loc[minutes.isin(minute_slots(frequency)) & frame.date.eq(frame.date.dt.floor("min"))].copy()


def anchors_from_history(frame):
    records = []
    for instrument, stock in regular(frame, "bar1m").sort_values(["instrument", "date"]).groupby("instrument", sort=False):
        previous, previous_day = np.nan, pd.NaT
        for day, group in stock.groupby(stock.date.dt.normalize(), sort=True):
            records.append({"instrument": instrument, "day": day, "anchor": previous, "anchor_day": previous_day})
            close = group.close.to_numpy(dtype=np.float64)
            valid = np.isfinite(close) & (close > 0)
            if valid.any(): previous, previous_day = float(close[valid][-1]), day
    return pd.DataFrame(records)


def temporal_close_counts(frame, frequency, price_parameters=None):
    slots = {int(value): i for i, value in enumerate(minute_slots(frequency))}
    price_parameters = price_parameters or {"center": 0., "scale": 1.}
    result = {key: 0 for key in ("valid_close_pairs", "nonzero_close_changes", "z_fp16_erased", "anchor_fp32_erased", "global_anchor_fp32_erased")}
    for _, group in frame.sort_values(["instrument", "date"]).groupby(["instrument", "day"], sort=False):
        raw = group.close.to_numpy(dtype=np.float64)
        z = rounded(group.z_close.to_numpy(dtype=np.float32))
        anchor = group.anchor.to_numpy(dtype=np.float64)
        valid = np.isfinite(raw) & (raw > 0) & np.isfinite(anchor) & (anchor > 0) & np.isfinite(z)
        if not (group.loc[valid, "anchor_day"] < group.loc[valid, "day"]).all():
            raise ValueError("价格锚点并非来自已观察的先前交易日")
        with np.errstate(invalid="ignore", divide="ignore"):
            encoded = (100 * (np.log(raw).astype(np.float32) - np.log(anchor).astype(np.float32))).astype(np.float32)
            global_encoded = ((np.log(raw).astype(np.float32) - np.float32(price_parameters["center"]))
                              / np.float32(price_parameters["scale"])).astype(np.float32)
        indices = np.array([slots[int(value)] for value in group.date.dt.hour * 60 + group.date.dt.minute])
        adjacent = valid[1:] & valid[:-1] & (np.diff(indices) == 1)
        changed = adjacent & (np.diff(raw) != 0)
        result["valid_close_pairs"] += int(adjacent.sum())
        result["nonzero_close_changes"] += int(changed.sum())
        result["z_fp16_erased"] += int((changed & (np.diff(z) == 0)).sum())
        result["anchor_fp32_erased"] += int((changed & (np.diff(encoded) == 0)).sum())
        result["global_anchor_fp32_erased"] += int((changed & (np.diff(global_encoded) == 0)).sum())
    return result


def frame_digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/quantity-coordinate-audit-20261023.json")
    args = parser.parse_args()
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    reference = ROOT / "outputs/runs/v1601-price-geometry-s42/data-manifest.json"
    manifest = read_json(reference)
    raw_root, prepared = Path(manifest["root"]), Path(manifest["preprocessing"]["train"]["path"])
    sources = {item["path"]: item for item in manifest["files"]}
    samples, parameter_records = [], {}
    previous_audit = read_json(ROOT / "experiments/price-geometry-audit-20261022.json")
    for year in (2020, 2021, 2022, 2023):
        month = f"{year}-06"
        universe = pd.read_parquet(raw_root / "cpt_jyc_2026_instruments.parquet", columns=KEYS,
            filters=[("date", ">=", pd.Timestamp(month + "-01")), ("date", "<", pd.Timestamp(f"{year}-07-01"))])
        days = pd.DatetimeIndex(sorted(universe.date.dt.normalize().unique())[:5])
        pool = sorted(universe.loc[universe.date.dt.normalize().eq(days[0]), "instrument"].unique())
        stocks = sorted(np.random.default_rng(np.random.SeedSequence([20261022, year])).choice(pool, size=min(64, len(pool)), replace=False).tolist())
        stock_digest = hashlib.sha256("\n".join(stocks).encode()).hexdigest()
        prior_sample = next(s for s in previous_audit["samples"] if s["month"] == month and s["frequency"] == "bar1m")
        assert stock_digest == prior_sample["stock_selection_sha256"]
        filters = [("date", ">=", days[0]), ("date", "<", days[-1] + pd.Timedelta(days=1)), ("instrument", "in", stocks)]
        one_minute = pd.read_parquet(raw_root / "cpt_jyc_2026_stock_bar1m" / (month + ".parquet"), columns=KEYS + ["close"] + FIELDS, filters=filters)
        anchors = anchors_from_history(one_minute)
        for frequency in ("bar1m", "bar5m", "bar30m"):
            raw_path = raw_root / ("cpt_jyc_2026_stock_" + frequency) / (month + ".parquet")
            raw = one_minute if frequency == "bar1m" else pd.read_parquet(raw_path, columns=KEYS + ["close"] + FIELDS, filters=filters)
            columns = ["z_close"] + [kind + "_" + field for kind in ("z", "c") for field in FIELDS]
            processed_path = prepared / "data" / frequency / (month + ".parquet")
            processed = pd.read_parquet(processed_path, columns=KEYS + columns, filters=filters)
            raw, processed = regular(raw, frequency), regular(processed, frequency)
            frame = raw.merge(processed, on=KEYS, how="outer", validate="one_to_one", indicator=True)
            if not frame._merge.eq("both").all():
                raise ValueError("原始与预处理样本键不同")
            frame["day"] = frame.date.dt.normalize()
            frame = frame.merge(anchors, on=["day", "instrument"], how="left", validate="many_to_one")
            frame = frame.loc[frame.day > days[0]].copy()
            statistics_path = prepared / "normalization" / (frequency + ".json")
            statistics = read_json(statistics_path)
            if pd.Timestamp(statistics["fit_interval"]["stop"]) > pd.Timestamp("2024-01-01"):
                raise ValueError("归一化参数越过固定训练区间")
            group_results = {}
            price_parameters = pooled_group_parameters(statistics, ["close"])
            parameter_records.setdefault(frequency, {})["price_anchor"] = price_parameters
            for name, fields in GROUPS.items():
                parameters = pooled_group_parameters(statistics, fields)
                parameter_records.setdefault(frequency, {})[name] = parameters
                values = frame[fields].to_numpy(dtype=np.float64)
                z = frame[["z_" + field for field in fields]].to_numpy(dtype=np.float32)
                c = frame[["c_" + field for field in fields]].to_numpy(dtype=np.float32)
                shared = common_quantity_view(values, parameters)
                views = {"z_fp32": z, "c_fp32": c, "z_fp16": rounded(z), "c_fp16": rounded(c),
                         "common_fp32": shared, "common_fp16": rounded(shared)}
                group_results[name] = {view: {pair: fractions(pair_counts(values, data, indices))
                    for pair, indices in PAIR_TYPES.items()} for view, data in views.items()}
            expected = sources["raw/" + raw_path.relative_to(raw_root).as_posix()]
            assert raw_path.stat().st_size == expected["bytes"]
            row = {"month": month, "frequency": frequency, "rows": len(frame), "stock_selection_sha256": stock_digest,
                "dates": [str(day.date()) for day in days[1:]], "warmup_date": str(days[0].date()),
                "groups": group_results, "temporal_close": fractions(temporal_close_counts(frame, frequency, price_parameters)),
                "statistics_sha256": sha256(statistics_path),
                "raw_source": {"path": raw_path.relative_to(ROOT).as_posix(), "bytes": raw_path.stat().st_size,
                    "mtime_ns": raw_path.stat().st_mtime_ns, "expected_sha256_from_parent_manifest": expected["sha256"],
                    "full_file_sha256_recomputed": False, "sample_content_sha256": frame_digest(raw)},
                "processed_sample_content_sha256": frame_digest(processed)}
            samples.append(row)
            print({"month": month, "frequency": frequency, "rows": len(frame)}, flush=True)
    aggregate = {}
    for frequency in ("bar1m", "bar5m", "bar30m"):
        rows = [s for s in samples if s["frequency"] == frequency]
        aggregate[frequency] = {"rows": sum(s["rows"] for s in rows), "groups": {}}
        for name in GROUPS:
            aggregate[frequency]["groups"][name] = {}
            for view in rows[0]["groups"][name]:
                aggregate[frequency]["groups"][name][view] = {}
                for pair in PAIR_TYPES:
                    counts = {key: sum(s["groups"][name][view][pair][key] for s in rows)
                              for key in rows[0]["groups"][name][view][pair] if not key.endswith("_fraction")}
                    aggregate[frequency]["groups"][name][view][pair] = fractions(counts)
        counts = {key: sum(s["temporal_close"][key] for s in rows)
                  for key in rows[0]["temporal_close"] if not key.endswith("_fraction")}
        aggregate[frequency]["temporal_close"] = fractions(counts)
    write_json(args.output, {"checked_at": now(), "script_sha256": sha256(__file__),
        "source_reference_run": reference.parent.name, "data_manifest_sha256": sha256(reference),
        "sample_protocol": "复用既有价格审计固定的2020—2023各六月首五日和每年64股；首日仅提供历史锚点。",
        "labels_read": False, "exposure_read": False, "cache_modified": False,
        "formal_training_performed": False, "new_candidate_created": False,
        "quantity_coordinates": "同类十档委托量或委托笔数的log1p值共用固定训练期全局中心与尺度；不混合成交量或金额单位。",
        "temporal_coordinates": "将当前已闭合Bar收盘价的log存为FP32，分别比较固定训练全局标准化和共同先前交易日中心；只统计相邻规则Bar数值变化，不构造收益标签。",
        "group_parameters": parameter_records, "pair_types": PAIR_TYPES, "aggregate": aggregate, "samples": samples,
        "limitations": ["归一化后的字段关系不同不等于数据错误或全部信息不可逆；该诊断不证明收益预测改善。",
            "没有默认要求不同盘口数量档位单调；只比较同一对提供字段的数值相对关系。",
            "价格时间路径比较的是数值变化是否被编码合并，不是经济信号强弱或可交易收益。",
            "仅训练期固定抽样，源清单哈希与样本内容散列分开记录，未重新计算整月文件哈希。",
            "此处比较固定共同坐标，没有验证任何可学习归一化模型。"]})


if __name__ == "__main__":
    main()
