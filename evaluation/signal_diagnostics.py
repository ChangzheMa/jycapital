"""预测信号的日内持续性与换手代理；不复刻平台 BARRA、组合或成本。"""

import numpy as np
import pandas as pd


def tail_weights(scores, fraction=0.1):
    """按分数选两端，各 0.5 名义权重；边界同分按相同比例分配。"""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
        raise ValueError("需要至少两个有限的一维分数")
    if not 0 < fraction < 0.5:
        raise ValueError("两端比例必须在 (0, 0.5) 内")
    count = max(1, int(np.floor(len(scores) * fraction)))
    low = np.partition(scores, count - 1)[count - 1]
    high = np.partition(scores, len(scores) - count)[len(scores) - count]

    def allocate(strict, tied):
        return (strict.astype(float) + tied * ((count - strict.sum()) / tied.sum())) / (2 * count)

    short = allocate(scores < low, scores == low)
    long = allocate(scores > high, scores == high)
    return long - short


def intraday_diagnostics(predictions, fraction=0.1, min_common=30):
    """只配对同一交易日相邻采样；持仓构造不读取标签。"""
    if list(predictions.columns) != ["date", "instrument", "factor"]:
        raise ValueError("预测列必须为 date、instrument、factor")
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame.date)
    if frame[["date", "instrument"]].isna().any().any() or frame.duplicated(["date", "instrument"]).any():
        raise ValueError("预测键为空或重复")
    if not np.isfinite(frame.factor).all():
        raise ValueError("预测分数非有限")
    records, previous = [], None
    for stamp, section in frame.sort_values(["date", "instrument"]).groupby("date", sort=True):
        scores = section.set_index("instrument").factor
        weights = pd.Series(tail_weights(scores.to_numpy(), fraction), index=scores.index)
        gross = float(weights.abs().sum())
        current = {"date": stamp, "scores": scores, "weights": weights, "gross": gross}
        if previous is not None and previous["date"].normalize() == stamp.normalize():
            old_weights, new_weights = previous["weights"].align(weights, join="outer", fill_value=0)
            old_scores, new_scores = previous["scores"].align(scores, join="inner")
            rank_correlation = np.nan
            if len(old_scores) >= min_common and old_scores.nunique() > 1 and new_scores.nunique() > 1:
                rank_correlation = old_scores.rank(method="average").corr(new_scores.rank(method="average"))
            portfolio_defined = bool(np.isclose(previous["gross"], 1) and np.isclose(gross, 1))
            records.append({"date": stamp, "previous_date": previous["date"], "common_stocks": len(old_scores),
                            "rank_autocorrelation": rank_correlation,
                            "one_way_turnover": float(0.5 * (new_weights - old_weights).abs().sum()) if portfolio_defined else np.nan,
                            "long_retention": float(np.minimum(new_weights.clip(lower=0), old_weights.clip(lower=0)).sum() / 0.5) if portfolio_defined else np.nan,
                            "short_retention": float(np.minimum((-new_weights).clip(lower=0), (-old_weights).clip(lower=0)).sum() / 0.5) if portfolio_defined else np.nan,
                            "portfolio_defined": portfolio_defined})
        previous = current
    transitions = pd.DataFrame(records, columns=["date", "previous_date", "common_stocks", "rank_autocorrelation",
                                                "one_way_turnover", "long_retention", "short_retention", "portfolio_defined"])
    if transitions.empty:
        raise ValueError("没有同一交易日内的相邻预测截面")
    fields = ["rank_autocorrelation", "one_way_turnover", "long_retention", "short_retention"]
    summary = {"transitions": len(transitions), "undefined_portfolio_transitions": int((~transitions.portfolio_defined).sum()),
               "undefined_rank_transitions": int(transitions.rank_autocorrelation.isna().sum()),
               "minimum_common_stocks": int(transitions.common_stocks.min())}
    summary.update({key + "_mean": float(transitions[key].mean()) if transitions[key].notna().any() else None for key in fields})
    return summary, transitions
