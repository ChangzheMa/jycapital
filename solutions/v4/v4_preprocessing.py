"""四频共用的逐字段变换；不切窗、不拟合模型、不读取标签。"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["date", "instrument"]
FREQUENCIES = ["bar1m", "bar5m", "bar15m", "bar30m"]
FIELDS = ["open", "high", "low", "close", "volume", "amount"] + [
    f"{prefix}{level}"
    for prefix in ["ask_price", "bid_price", "ask_volume", "bid_volume", "ask_num_orders", "bid_num_orders"]
    for level in range(1, 6)
]
PRICE_FIELDS = {"open", "high", "low", "close"} | {
    f"{side}_price{level}" for side in ["ask", "bid"] for level in range(1, 6)
}


def output_columns(fields=FIELDS):
    return KEYS + [f"{kind}_{field}" for kind in ["z", "c"] for field in fields]


def log_values(frame, fields=FIELDS):
    """零价格无效；零数量有效。缺失保留为 NaN，稍后写为 Parquet null。"""
    raw = frame[list(fields)].to_numpy(dtype=np.float64, copy=True)
    raw[~np.isfinite(raw)] = np.nan
    prices = np.array([field in PRICE_FIELDS for field in fields])
    raw[:, prices] = np.where(raw[:, prices] > 0, raw[:, prices], np.nan)
    raw[:, ~prices] = np.where(raw[:, ~prices] >= 0, raw[:, ~prices], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw[:, prices] = np.log(raw[:, prices])
        raw[:, ~prices] = np.log1p(raw[:, ~prices])
    return raw


def empty_moments(width):
    return {"count": np.zeros(width, dtype=np.int64), "mean": np.zeros(width), "m2": np.zeros(width)}


def unpack_moments(value):
    result = {"count": np.asarray(value["count"], dtype=np.int64),
              "mean": np.asarray(value["mean"], dtype=np.float64),
              "m2": np.asarray(value["m2"], dtype=np.float64)}
    if np.any(result["count"] < 0) or np.any(result["m2"] < 0):
        raise ValueError("统计量计数或离差平方和非法")
    if not all(np.isfinite(array).all() for array in result.values()):
        raise ValueError("统计量必须有限")
    return result


def pack_moments(value):
    return {key: array.tolist() for key, array in value.items()}


def merge_moments(left, right):
    """Chan 合并公式，避免直接使用大数平方和相减估计方差。"""
    count = left["count"] + right["count"]
    divisor = np.maximum(count, 1)
    delta = right["mean"] - left["mean"]
    mean = left["mean"] + delta * right["count"] / divisor
    m2 = left["m2"] + right["m2"] + delta ** 2 * left["count"] * right["count"] / divisor
    return {"count": count, "mean": np.where(count > 0, mean, 0.0), "m2": np.maximum(m2, 0.0)}


def moments_from_values(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    mean = np.where(valid, values, 0).sum(axis=0) / np.maximum(count, 1)
    centered = np.where(valid, values - mean, 0)
    return {"count": count, "mean": mean, "m2": np.square(centered).sum(axis=0)}


def population_std(value):
    return np.sqrt(value["m2"] / np.maximum(value["count"], 1))


def running_moments(values, previous):
    """含当前已结束 Bar 的前缀统计；返回各行统计及最终状态。"""
    valid = np.isfinite(values)
    # 按列选择一个平移原点改善数值精度，不改变任何前缀的统计结果。
    first = np.argmax(valid, axis=0)
    origin = np.where(previous["count"] > 0, previous["mean"],
                      np.where(valid.any(axis=0), values[first, np.arange(values.shape[1])], 0.0))
    delta = np.where(valid, values - origin, 0.0)
    count = np.cumsum(valid, axis=0, dtype=np.int64) + previous["count"]
    total = np.cumsum(delta, axis=0) + previous["count"] * (previous["mean"] - origin)
    square = (np.cumsum(delta ** 2, axis=0) + previous["m2"]
              + previous["count"] * (previous["mean"] - origin) ** 2)
    divisor = np.maximum(count, 1)
    mean = origin + total / divisor
    m2 = np.maximum(square - total ** 2 / divisor, 0.0)
    mean = np.where(count > 0, mean, 0.0)
    final = {"count": count[-1].copy(), "mean": mean[-1].copy(), "m2": m2[-1].copy()}
    return count, mean, np.sqrt(m2 / divisor), final


class UniverseIndex:
    """Bar 时点采用不晚于该时点的最后一份完整股票池快照。"""

    def __init__(self, universe):
        if universe[KEYS].isna().any().any() or universe.duplicated(KEYS).any():
            raise ValueError("股票池存在空键或重复键")
        self.times, self.pools = [], []
        previous = None
        for timestamp, frame in universe.groupby("date", sort=True, observed=True):
            pool = frozenset(frame["instrument"].astype(str))
            if pool != previous:
                self.times.append(pd.Timestamp(timestamp).value)
                self.pools.append(pool)
                previous = pool
        self.times = np.array(self.times, dtype=np.int64)

    def eligible(self, frame):
        times = pd.to_datetime(frame["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        ids = np.searchsorted(self.times, times, side="right") - 1
        stocks = frame["instrument"].astype(str).to_numpy()
        result = np.zeros(len(frame), dtype=bool)
        for index in np.unique(ids):
            if index < 0:
                continue
            positions = np.flatnonzero(ids == index)
            pool = self.pools[index]
            result[positions] = np.fromiter((stocks[p] in pool for p in positions), dtype=bool, count=len(positions))
        return result


class Normalizer:
    """固定训练统计 + 缺参时的因果历史回退 + 同时点截面统计。

    每次调用必须包含完整且按时间递增的截面。非训练调用需从训练期之后
    顺序输入全部可用历史；history_dict 可保存／恢复动态回退状态。
    """

    def __init__(self, statistics, config, *, training=False, history_dict=None):
        self.fields = list(statistics["fields"])
        self.minimum = config["normalization"]["min_observations"]
        self.epsilon = config["normalization"]["scale_epsilon"]
        self.cs_minimum = config["normalization"]["min_cross_section_observations"]
        self.training = training
        self.fit_stop = pd.Timestamp(statistics["fit_interval"]["stop"])
        self.stocks = {key: unpack_moments(value) for key, value in statistics["stocks"].items()}
        self.global_moments = unpack_moments(statistics["global"])
        self.global_std = population_std(self.global_moments)
        # 训练期全体股票也恒定时，以 1 为尺度，仍保留相对训练均值的变化。
        self.global_scale = np.where(self.global_std > self.epsilon, self.global_std, 1.0)
        self.history = {key: unpack_moments(value) for key, value in (history_dict or {}).get("stocks", {}).items()}
        self.last_time = pd.Timestamp(history_dict["last_time"]) if history_dict and history_dict.get("last_time") else None

    def history_dict(self):
        return {"last_time": str(self.last_time) if self.last_time is not None else None,
                "stocks": {key: pack_moments(value) for key, value in sorted(self.history.items())}}

    def transform(self, frame, eligible, *, logged=False):
        frame = frame.reset_index(drop=True)
        if frame.empty:
            return pd.DataFrame(columns=output_columns(self.fields)), {}
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError("行情存在空键或重复键")
        dates = pd.to_datetime(frame["date"])
        if not dates.is_monotonic_increasing:
            raise ValueError("处理输入必须按时间递增")
        if self.last_time is not None and dates.iloc[0] <= self.last_time:
            raise ValueError("截面不能重复或跨块拆分，历史必须顺序处理")
        if not self.training and dates.iloc[0] < self.fit_stop:
            raise ValueError("冻结统计量只能用于训练期之后；训练期转换需显式 training=True")
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != (len(frame),):
            raise ValueError("股票池掩码长度不匹配")
        values = frame[self.fields].to_numpy(dtype=np.float64) if logged else log_values(frame, self.fields)
        if np.isinf(values).any():
            raise ValueError("对数输入包含无穷值")
        codes, stocks = pd.factorize(frame["instrument"].astype(str), sort=False)
        width = len(self.fields)
        stock_moments = [self.stocks.get(stock, empty_moments(width)) for stock in stocks]
        means = np.stack([item["mean"] for item in stock_moments])
        scales = np.stack([population_std(item) for item in stock_moments])
        trusted = np.stack([item["count"] >= self.minimum for item in stock_moments]) & (scales > self.epsilon)
        z = (values - means[codes]) / np.where(trusted, scales, 1.0)[codes]
        finite = np.isfinite(values)
        quality = {"rows": len(frame), "missing_input_values": int((~finite).sum()),
                   "fixed_values": int((finite & trusted[codes]).sum()), "dynamic_values": 0,
                   "pooled_values": 0, "unavailable_values": 0, "eligible_rows": int(eligible.sum())}
        for code in np.flatnonzero(~trusted.all(axis=1)):
            positions = np.flatnonzero(codes == code)
            stock = stocks[code]
            fallback_fields = ~trusted[code]
            previous = self.history.get(stock)
            if previous is None:
                previous = empty_moments(width) if self.training else stock_moments[code]
            counts, running_mean, running_std, final = running_moments(values[positions], previous)
            self.history[stock] = final
            dynamic = (counts >= self.minimum) & (running_std > self.epsilon)
            pooled = (~dynamic) & (self.global_moments["count"] >= self.minimum)
            replacement = np.where(dynamic,
                (values[positions] - running_mean) / np.where(dynamic, running_std, 1.0),
                (values[positions] - self.global_moments["mean"]) / self.global_scale)
            replacement[~(dynamic | pooled)] = np.nan
            z[np.ix_(positions, np.flatnonzero(fallback_fields))] = replacement[:, fallback_fields]
            valid_fallback = finite[positions] & fallback_fields
            quality["dynamic_values"] += int((valid_fallback & dynamic).sum())
            quality["pooled_values"] += int((valid_fallback & pooled).sum())
            quality["unavailable_values"] += int((valid_fallback & ~(dynamic | pooled)).sum())

        # reduceat 同时对所有字段计算每个完整时间截面的统计。
        timestamps = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(timestamps)) + 1]
        sizes = np.diff(np.r_[starts, len(frame)])
        valid_cross = np.isfinite(z) & eligible[:, None]
        count = np.add.reduceat(valid_cross.astype(np.int64), starts, axis=0)
        sums = np.add.reduceat(np.where(valid_cross, z, 0.0), starts, axis=0)
        cross_mean = sums / np.maximum(count, 1)
        centered = z - np.repeat(cross_mean, sizes, axis=0)
        square = np.add.reduceat(np.where(valid_cross, centered ** 2, 0.0), starts, axis=0)
        cross_std = np.sqrt(square / np.maximum(count, 1))
        usable = count >= self.cs_minimum
        scale = np.repeat(np.where(cross_std > self.epsilon, cross_std, 1.0), sizes, axis=0)
        c = centered / scale
        c[~np.repeat(usable, sizes, axis=0)] = np.nan
        output_values = np.concatenate([z, c], axis=1)
        if np.isinf(output_values).any() or np.any(np.abs(output_values[np.isfinite(output_values)]) > np.finfo(np.float32).max):
            raise FloatingPointError("归一化结果超出 float32 有限范围")
        result = pd.DataFrame(output_values.astype(np.float32), columns=output_columns(self.fields)[2:])
        result.insert(0, "instrument", frame["instrument"].astype(str))
        result.insert(0, "date", dates)
        quality["missing_output_values"] = int(np.isnan(output_values).sum())
        quality["cross_sections"] = len(starts)
        quality["degenerate_cross_section_fields"] = int((usable & (cross_std <= self.epsilon)).sum())
        self.last_time = dates.iloc[-1]
        return result, quality
