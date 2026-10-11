"""所有方案共用的本地评估器，不模拟未公开的 BARRA/平台评分。"""

import numpy as np
import pandas as pd

KEYS = ["date", "instrument"]


def _correlation(x, y, groups):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    dx = x - x.groupby(groups).transform("mean")
    dy = y - y.groupby(groups).transform("mean")
    covariance = (dx * dy).groupby(groups).sum()
    denominator = np.sqrt((dx * dx).groupby(groups).sum() * (dy * dy).groupby(groups).sum())
    return covariance / denominator.where(denominator > 0)


def evaluate(predictions, universe_labels, protocol):
    if list(predictions.columns) != KEYS + ["factor"]:
        raise ValueError("输出列必须恰好为 date, instrument, factor")
    if predictions.duplicated(KEYS).any() or predictions[KEYS].isna().any().any():
        raise ValueError("预测包含重复或空键")
    if universe_labels.duplicated(KEYS).any():
        raise ValueError("评估股票池包含重复键")
    if not np.isfinite(predictions["factor"].to_numpy(dtype=float)).all():
        raise ValueError("预测包含非有限值")
    extra = predictions[KEYS].merge(universe_labels[KEYS], on=KEYS, how="left", indicator=True)
    if (extra["_merge"] != "both").any():
        raise ValueError("预测包含股票池以外的键")
    label = protocol["label_column"]
    frame = universe_labels[KEYS + [label]].merge(predictions, on=KEYS, how="left", validate="one_to_one")
    finite_label = np.isfinite(frame[label].to_numpy(dtype=float))
    available = frame["factor"].notna()
    by_date = frame.groupby("date").size().rename("expected").to_frame()
    by_date["predicted"] = available.groupby(frame["date"]).sum()
    by_date["coverage"] = by_date["predicted"] / by_date["expected"]
    by_date["valid_labels"] = pd.Series(finite_label, index=frame.index).groupby(frame["date"]).sum()
    valid = frame.loc[finite_label & available].copy()
    groups = valid["date"]
    by_date["evaluated"] = valid.groupby("date").size().reindex(by_date.index, fill_value=0)
    by_date["ic"] = _correlation(valid["factor"], valid[label], groups)
    rank_x = valid.groupby("date")["factor"].rank(method="average")
    rank_y = valid.groupby("date")[label].rank(method="average")
    by_date["rank_ic"] = _correlation(rank_x, rank_y, groups)
    too_small = by_date["evaluated"] < protocol["min_cross_section_size"]
    by_date.loc[too_small, ["ic", "rank_ic"]] = np.nan
    squared = (valid["factor"] - valid[label]) ** 2
    by_date["mse"] = squared.groupby(groups).mean()
    stats = {
        "protocol_id": protocol["protocol_id"],
        "expected_rows": len(frame),
        "predicted_rows": int(available.sum()),
        "valid_label_rows": int(finite_label.sum()),
        "evaluated_rows": len(valid),
        "coverage": float(available.mean()),
        "min_cross_section_coverage": float(by_date["coverage"].min()),
        "coverage_violations": int((by_date["coverage"] < 1 - protocol["max_missing_fraction_per_cross_section"]).sum()),
        "cross_sections": len(by_date),
        "mse": float(squared.mean()),
        "mae": float((valid["factor"] - valid[label]).abs().mean()),
        "zero_prediction_mse": float((valid[label] ** 2).mean()),
    }
    for metric in ["ic", "rank_ic"]:
        series = by_date[metric].dropna()
        std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
        stats[f"{metric}_mean"] = float(series.mean()) if len(series) else None
        stats[f"{metric}_std"] = std if np.isfinite(std) else None
        stats[f"{metric}_ir"] = float(series.mean() / std) if std > 0 else None
        stats[f"{metric}_valid_cross_sections"] = len(series)
        stats[f"{metric}_undefined_cross_sections"] = len(by_date) - len(series)
    monthly = by_date.groupby(by_date.index.to_period("M")).agg(
        ic_mean=("ic", "mean"), rank_ic_mean=("rank_ic", "mean"),
        mean_coverage=("coverage", "mean"), min_coverage=("coverage", "min"),
        cross_sections=("expected", "size"), evaluated_rows=("evaluated", "sum"),
    )
    monthly.index = monthly.index.astype(str)
    return stats, by_date.reset_index(), monthly.reset_index(names="month")
