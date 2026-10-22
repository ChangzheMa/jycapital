"""只读训练期价格输入，检查逐字段归一化与共享价格中心的几何关系。"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in ("common", "solutions/v1002"):
    sys.path.insert(0, str(ROOT / folder))
from artifacts import now, read_json, sha256, write_json
from v1002_data import minute_slots

FIELDS = ["open", "high", "low", "close"] + [
    f"{side}_price{level}" for side in ("ask", "bid") for level in range(1, 6)
]
PAIR_GROUPS = {
    "ask_ladder": [(f"ask_price{i}", f"ask_price{i + 1}") for i in range(1, 5)],
    "bid_ladder": [(f"bid_price{i + 1}", f"bid_price{i}") for i in range(1, 5)],
    "best_bid_ask": [("bid_price1", "ask_price1")],
    "ohlc": [("low", "open"), ("low", "close"), ("open", "high"), ("close", "high")],
}


def regular(frame, frequency):
    minute = frame.date.dt.hour * 60 + frame.date.dt.minute
    return frame.loc[minute.isin(minute_slots(frequency)) &
                     frame.date.eq(frame.date.dt.floor("min"))].copy()


def prior_close_anchors(frame):
    """同一股票当天只能使用此前已观察日的最后有限正收盘价。"""
    frame = regular(frame, "bar1m").sort_values(["instrument", "date"])
    rows = []
    for instrument, part in frame.groupby("instrument", sort=False):
        anchor, anchor_day = np.nan, pd.NaT
        for day, group in part.groupby(part.date.dt.normalize(), sort=True):
            rows.append({"day": day, "instrument": instrument,
                         "anchor": anchor, "anchor_day": anchor_day})
            values = group.close.to_numpy(dtype=np.float64)
            usable = np.isfinite(values) & (values > 0)
            if usable.any():
                anchor, anchor_day = float(values[usable][-1]), day
    return pd.DataFrame(rows, columns=["day", "instrument", "anchor", "anchor_day"])


def rounded(values):
    with np.errstate(over="ignore", invalid="ignore"):
        half = values.astype(np.float16).astype(np.float32)
    return np.where(np.isfinite(values) & ~np.isfinite(half), values, half)


def shared_center(prices, anchors):
    prices, anchors = np.asarray(prices, np.float64), np.asarray(anchors, np.float64)
    valid = np.isfinite(prices) & (prices > 0) & np.isfinite(anchors[:, None]) & (anchors[:, None] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        centered = 100 * (np.log(prices) - np.log(anchors[:, None]))
    return np.where(valid, centered, np.nan).astype(np.float32)


def relation_counts(raw, transformed, pairs):
    result = {key: 0 for key in (
        "valid_pairs", "raw_order_violations", "strict_raw_pairs", "strict_reversed",
        "strict_erased", "raw_ties", "raw_ties_split", "sign_agreement_pairs"
    )}
    for left, right in pairs:
        a, b = FIELDS.index(left), FIELDS.index(right)
        valid = (np.isfinite(raw[:, a]) & np.isfinite(raw[:, b]) &
                 (raw[:, a] > 0) & (raw[:, b] > 0) &
                 np.isfinite(transformed[:, a]) & np.isfinite(transformed[:, b]))
        delta = raw[valid, b].astype(np.float64) - raw[valid, a].astype(np.float64)
        actual = transformed[valid, b].astype(np.float64) - transformed[valid, a].astype(np.float64)
        strict, tied = delta > 0, delta == 0
        result["valid_pairs"] += int(valid.sum())
        result["raw_order_violations"] += int((delta < 0).sum())
        result["strict_raw_pairs"] += int(strict.sum())
        result["strict_reversed"] += int((strict & (actual < 0)).sum())
        result["strict_erased"] += int((strict & (actual == 0)).sum())
        result["raw_ties"] += int(tied.sum())
        result["raw_ties_split"] += int((tied & (actual != 0)).sum())
        result["sign_agreement_pairs"] += int((np.sign(delta) == np.sign(actual)).sum())
    return result


def summarize(counts):
    result = dict(counts)
    for numerator, denominator in (("strict_reversed", "strict_raw_pairs"),
                                   ("strict_erased", "strict_raw_pairs"),
                                   ("raw_ties_split", "raw_ties"),
                                   ("raw_order_violations", "valid_pairs")):
        result[numerator + "_fraction"] = counts[numerator] / counts[denominator] if counts[denominator] else None
    return result


def inspect_frames(raw, processed, anchors, frequency, first_day):
    raw, processed = regular(raw, frequency), regular(processed, frequency)
    merged = raw.merge(processed, on=["date", "instrument"], how="outer", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("原始行情与冻结预处理键不一致，不能静默筛掉缺失行")
    merged["day"] = merged.date.dt.normalize()
    merged = merged.merge(anchors, on=["day", "instrument"], how="left", validate="many_to_one")
    scored = merged.day > first_day
    anchored = scored & np.isfinite(merged.anchor) & (merged.anchor > 0) & (merged.anchor_day < merged.day)
    part = merged.loc[anchored]
    raw_prices = part[FIELDS].to_numpy(dtype=np.float64)
    z = part[["z_" + field for field in FIELDS]].to_numpy(dtype=np.float32)
    c = part[["c_" + field for field in FIELDS]].to_numpy(dtype=np.float32)
    common = shared_center(raw_prices, part.anchor.to_numpy())
    views = {"raw": raw_prices, "z_fp32": z, "c_fp32": c, "z_fp16": rounded(z),
             "c_fp16": rounded(c), "shared_center_fp32": common, "shared_center_fp16": rounded(common)}
    return {
        "regular_rows": len(merged), "scored_rows": int(scored.sum()),
        "anchored_rows": int(anchored.sum()), "unavailable_anchor_rows": int((scored & ~anchored).sum()),
        "groups": {name: {group: summarize(relation_counts(raw_prices, values, pairs))
                          for group, pairs in PAIR_GROUPS.items()} for name, values in views.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/price-geometry-audit-20261022.json")
    args = parser.parse_args()
    reference = ROOT / "outputs/runs/v1002-current-temporal-experts-s42/data-manifest.json"
    manifest = read_json(reference)
    data_root = Path(manifest["root"])
    preprocessing = Path(manifest["preprocessing"]["train"]["path"])
    samples = []
    for year in (2020, 2021, 2022, 2023):
        month = f"{year}-06"
        universe = pd.read_parquet(data_root / "cpt_jyc_2026_instruments.parquet", columns=["date", "instrument"],
                                   filters=[("date", ">=", pd.Timestamp(month + "-01")),
                                            ("date", "<", pd.Timestamp(f"{year}-07-01"))])
        days = pd.DatetimeIndex(sorted(universe.date.dt.normalize().unique())[:5])
        pool = sorted(universe.loc[universe.date.dt.normalize().eq(days[0]), "instrument"].unique())
        stocks = sorted(np.random.default_rng(np.random.SeedSequence([20261022, year])).choice(
            pool, size=min(64, len(pool)), replace=False).tolist())
        filters = [("date", ">=", days[0]), ("date", "<", days[-1] + pd.Timedelta(days=1)), ("instrument", "in", stocks)]
        one_minute = pd.read_parquet(data_root / "cpt_jyc_2026_stock_bar1m" / (month + ".parquet"),
                                     columns=["date", "instrument", *FIELDS], filters=filters)
        anchors = prior_close_anchors(one_minute)
        for frequency in ("bar1m", "bar5m", "bar30m"):
            raw_path = data_root / ("cpt_jyc_2026_stock_" + frequency) / (month + ".parquet")
            processed_path = preprocessing / "data" / frequency / (month + ".parquet")
            raw = one_minute if frequency == "bar1m" else pd.read_parquet(raw_path, columns=["date", "instrument", *FIELDS], filters=filters)
            processed = pd.read_parquet(processed_path, columns=["date", "instrument"] +
                                        [kind + "_" + field for kind in ("z", "c") for field in FIELDS], filters=filters)
            result = inspect_frames(raw, processed, anchors, frequency, days[0])
            result.update(month=month, frequency=frequency, stocks=len(stocks), warmup_date=str(days[0].date()),
                          stock_selection_sha256=hashlib.sha256("\n".join(stocks).encode()).hexdigest(),
                          dates=[str(day.date()) for day in days[1:]],
                          raw_source={"path": raw_path.relative_to(ROOT).as_posix(), "bytes": raw_path.stat().st_size,
                                      "mtime_ns": raw_path.stat().st_mtime_ns},
                          processed_record=read_json(preprocessing / "records" / frequency / (month + ".json"))["sha256"])
            samples.append(result)
            print({"month": month, "frequency": frequency, "rows": result["anchored_rows"],
                   "z_spread_reversed": result["groups"]["z_fp32"]["best_bid_ask"]["strict_reversed_fraction"]}, flush=True)
    aggregate = {}
    for frequency in ("bar1m", "bar5m", "bar30m"):
        selected = [item for item in samples if item["frequency"] == frequency]
        aggregate[frequency] = {}
        for view in selected[0]["groups"]:
            aggregate[frequency][view] = {}
            for group in PAIR_GROUPS:
                counts = {k: sum(item["groups"][view][group][k] for item in selected)
                          for k in selected[0]["groups"][view][group] if not k.endswith("_fraction")}
                aggregate[frequency][view][group] = summarize(counts)
    write_json(args.output, {"schema_version": 1, "checked_at": now(), "source_reference_run": reference.parent.name,
                            "data_manifest_sha256": sha256(reference), "script_sha256": sha256(__file__),
                            "scope": "2020—2023 年各六月首五个交易日，首日池确定随机64股；仅规则 Bar，首日暖启动，缺历史锚点显式报告。",
                            "labels_read": False, "exposure_read": False, "cache_modified": False,
                            "formal_training_performed": False, "new_candidate_created": False,
                            "shared_center": "所有价格字段共享此前已观察交易日的最后有效1分钟收盘价；100*(log(price)-log(anchor))，不构造新收益标签。",
                            "pair_groups": PAIR_GROUPS, "aggregate": aggregate, "samples": samples,
                            "limitations": ["几何关系不直接保留不等于信息不可逆；提供完整统计量时可反变换，当前模型未接收这些统计量。",
                                            "相同价格中心仅是输入表达诊断，没有正式训练或预测改善证据。",
                                            "排除原始无效价格并单独统计原始次序异常；未以任何标签过滤样本。",
                                            "不假定盘口与OHLC在同一时刻采样；不将价差或其他人工统计作为模型输入。",
                                            "保持既有 vwap_return 和所有历史模型不变；没有读取或恢复 exposure。"]})


if __name__ == "__main__":
    main()
