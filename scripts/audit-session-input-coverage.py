"""对照冻结日内指标与训练期盘前行情覆盖；不训练、不改标签或输入缓存。"""
import argparse
import hashlib
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in ("common", "solutions/v1102"):
    sys.path.insert(0, str(ROOT / folder))
from artifacts import now, read_json, sha256, write_json
from v1102_data import minute_slots
from v1102_schema import NON_BOOK_FIELDS, BOOK_FIELDS

VERSIONS = ("v1002", "v1101", "v1102", "v1301", "v1401")
KEYS = ["date", "instrument"]
FIELDS = list(NON_BOOK_FIELDS + BOOK_FIELDS)
PRICE_FIELDS = ["open", "high", "low", "close"] + [
    f"{side}_price{level}" for side in ("ask", "bid") for level in range(1, 6)
]


def clock_metrics():
    result = []
    for version in VERSIONS:
        summary = read_json(ROOT / f"experiments/{version}-gru.json")
        for split in ("validation", "test"):
            path = ROOT / summary["run_path"] / "evaluation" / split / "cross-sections.parquet"
            frame = pd.read_parquet(path)
            if frame.date.duplicated().any():
                raise ValueError("冻结截面日期不唯一")
            frame["clock"] = frame.date.dt.strftime("%H:%M")
            frame["month"] = frame.date.dt.strftime("%Y-%m")
            rows = []
            for clock, group in frame.groupby("clock", sort=True):
                rows.append({"clock": clock, "cross_sections": len(group),
                             **{key: float(group[column].mean()) for key, column in
                                (("ic_mean", "ic"), ("rank_ic_mean", "rank_ic"))}})
            for key in ("ic_mean", "rank_ic_mean"):
                reconstructed = sum(row[key] * row["cross_sections"] for row in rows) / len(frame)
                if not np.isclose(reconstructed, summary["metrics"][split][key], rtol=0, atol=1e-12):
                    raise ValueError("分时点指标未还原冻结整体均值")
            monthly = []
            for month, group in frame.groupby("month", sort=True):
                opening, rest = group[group.clock.eq("09:30")], group[~group.clock.eq("09:30")]
                monthly.append({"month": month, "opening_sections": len(opening), "other_sections": len(rest),
                                "opening_ic": float(opening.ic.mean()), "other_ic": float(rest.ic.mean()),
                                "opening_rank_ic": float(opening.rank_ic.mean()), "other_rank_ic": float(rest.rank_ic.mean())})
            result.append({"solution_id": version, "run_id": summary["run_id"], "split": split,
                           "cross_sections_sha256": sha256(path), "clock_metrics": rows,
                           "opening_fraction": float(frame.clock.eq("09:30").mean()),
                           "other_clock_ic_mean": float(frame.loc[~frame.clock.eq("09:30"), "ic"].mean()),
                           "other_clock_rank_ic_mean": float(frame.loc[~frame.clock.eq("09:30"), "rank_ic"].mean()),
                           "monthly": monthly, "frozen_mean_reconstructed": True})
    return result


def raw_coverage():
    reference = ROOT / "outputs/runs/v1102-structured-temporal-experts-s42/data-manifest.json"
    manifest = read_json(reference)
    data_root, preprocessing = Path(manifest["root"]), Path(manifest["preprocessing"]["train"]["path"])
    records = []
    for year in (2020, 2021, 2022, 2023):
        month = f"{year}-06"
        universe = pd.read_parquet(data_root / "cpt_jyc_2026_instruments.parquet", columns=KEYS,
            filters=[("date", ">=", pd.Timestamp(month + "-01")), ("date", "<", pd.Timestamp(f"{year}-07-01"))])
        days = pd.DatetimeIndex(sorted(universe.date.dt.normalize().unique())[:5])
        opening = universe[universe.date.dt.strftime("%H:%M").eq("09:30")].copy()
        pool = sorted(opening.loc[opening.date.dt.normalize().eq(days[0]), "instrument"].unique())
        stocks = sorted(np.random.default_rng(np.random.SeedSequence([20261022, year])).choice(pool, size=64, replace=False).tolist())
        opening = opening[opening.instrument.isin(stocks) & opening.date.dt.normalize().isin(days)].copy()
        opening["day"] = opening.date.dt.normalize()
        eligible = opening[["day", "instrument"]]
        if eligible.duplicated().any():
            raise ValueError("股票池存在重复股票日")
        filters = [[("date", ">=", day + pd.Timedelta(hours=9)),
                    ("date", "<=", day + pd.Timedelta(hours=9, minutes=30)),
                    ("instrument", "in", stocks)] for day in days]
        for frequency in ("bar1m", "bar5m", "bar30m"):
            path = data_root / ("cpt_jyc_2026_stock_" + frequency) / (month + ".parquet")
            raw = pd.read_parquet(path, columns=KEYS + FIELDS, filters=filters)
            if raw.duplicated(KEYS).any():
                raise ValueError("原始行情键不唯一")
            raw["day"] = raw.date.dt.normalize()
            raw = raw.merge(eligible, on=["day", "instrument"], how="inner", validate="many_to_one")
            sample_hash = hashlib.sha256(pd.util.hash_pandas_object(
                raw[KEYS + FIELDS].sort_values(KEYS), index=False).to_numpy().tobytes()).hexdigest()
            prices = raw[PRICE_FIELDS].to_numpy(dtype=np.float64)
            raw["any_price"] = (np.isfinite(prices) & (prices > 0)).any(axis=1)
            raw["both_best_quotes"] = np.isfinite(raw.ask_price1) & raw.ask_price1.gt(0) & np.isfinite(raw.bid_price1) & raw.bid_price1.gt(0)
            raw["positive_volume"] = np.isfinite(raw.volume) & raw.volume.gt(0)
            raw["clock"] = raw.date.dt.strftime("%H:%M")
            admitted = (raw.date.dt.hour * 60 + raw.date.dt.minute).isin(minute_slots(frequency)) & raw.date.eq(raw.date.dt.floor("min"))
            processed_path = preprocessing / "data" / frequency / (month + ".parquet")
            normalized_columns = ["z_" + field for field in PRICE_FIELDS]
            processed = pd.read_parquet(processed_path, columns=KEYS + normalized_columns, filters=filters)
            matched = raw[KEYS].merge(processed, on=KEYS, how="inner", validate="one_to_one")
            per_day = raw.groupby(["day", "instrument"])[["any_price", "both_best_quotes", "positive_volume"]].any()
            row = {"month": month, "frequency": frequency, "dates": [str(day.date()) for day in days],
                   "stocks_selected": len(stocks), "eligible_stock_days": len(eligible),
                   "stock_selection_sha256": hashlib.sha256("\n".join(stocks).encode()).hexdigest(),
                   "raw_sample_sha256": sample_hash, "raw_sample_hash_method": "pandas.hash_pandas_object(sorted date/instrument, original 36 fields, index=False) then sha256",
                   "pandas_version": pd.__version__, "raw_source": {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns},
                   "processed_record_sha256": read_json(preprocessing / "records" / frequency / (month + ".json"))["sha256"],
                   "raw_rows": len(raw), "observed_stock_days": len(per_day),
                   "any_price_stock_days": int(per_day.any_price.sum()), "both_best_quotes_stock_days": int(per_day.both_best_quotes.sum()),
                   "positive_volume_stock_days": int(per_day.positive_volume.sum()),
                   "current_grid_admitted_rows": int(admitted.sum()), "current_grid_omitted_rows": int((~admitted).sum()),
                   "processed_matching_rows": len(matched),
                   "processed_any_finite_price_rows": int(np.isfinite(matched[normalized_columns].to_numpy()).any(axis=1).sum()),
                   "by_clock": [{"clock": clock, "rows": len(group), "any_price_rows": int(group.any_price.sum()),
                                 "both_best_quotes_rows": int(group.both_best_quotes.sum()), "positive_volume_rows": int(group.positive_volume.sum())}
                                for clock, group in raw.groupby("clock", sort=True)]}
            records.append(row)
            print({key: row[key] for key in ("month", "frequency", "raw_rows", "any_price_stock_days", "current_grid_admitted_rows")}, flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/session-input-audit-20261022.json")
    args = parser.parse_args()
    result = {"checked_at": now(), "script_sha256": sha256(__file__), "versions_fixed_before_analysis": list(VERSIONS),
              "source_files": {name: sha256(ROOT / name) for name in ("solutions/v1102/v1102_data.py", "solutions/v1102/v1102_schema.py")},
              "source_scope": "冻结验证/测试截面指标；原始行情抽样限2020—2023各年六月首五个交易日，首日09:30股票池确定随机64股，每日再匹配当时09:30成员。",
              "raw_time_interval": "各交易日09:00至09:30闭区间；未读取之后的原始行情作为该时点输入证据。",
              "label_table_read": False, "exposure_read": False, "cache_modified": False,
              "new_model_constructed": False, "formal_training_performed": False,
              "current_grid_first_bar": {frequency: f"{int(minute_slots(frequency)[0]) // 60:02d}:{int(minute_slots(frequency)[0]) % 60:02d}"
                                         for frequency in ("bar1m", "bar5m", "bar30m")},
              "clock_results": clock_metrics(), "input_samples": raw_coverage(),
              "limitations": ["盘前观测存在且未直接进入固定网格，不等于具有未来收益预测价值，也不是归一化实现错误。",
                              "原始输入覆盖仅为20个训练日及每年64股的抽样，不外推为全量或平台覆盖。",
                              "只比较原始字段可用性，不将其解释为已核实的逐笔委托含义；零成交量不自动判定价格或盘口无效。",
                              "分时点结果来自已观察研究区间，不用于事后挑时点、更换模型输出或缩减覆盖。",
                              "未重算或反推vwap_return；所有正式标签、股票池及横向评估口径不变。"]}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
