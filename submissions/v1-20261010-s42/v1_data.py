"""v1 专属的因果切窗及归一化；训练和推理共用此实现。"""

from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd

KEYS = ["date", "instrument"]


def local_time(value):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError("时间不能为空")
    if result.tzinfo is not None:
        result = result.tz_convert("Asia/Shanghai").tz_localize(None)
    return result


def platform_interval(start_date, end_date):
    """平台 end_date 为包含端点；内部统一使用半开区间。"""
    start, end = local_time(start_date), local_time(end_date)
    date_only = (
        isinstance(end_date, str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_date.strip()) is not None
    ) or (isinstance(end_date, date) and not isinstance(end_date, datetime))
    stop = end.normalize() + pd.Timedelta(days=1) if date_only else end.floor("min") + pd.Timedelta(minutes=1)
    if start >= stop or (not date_only and start > end):
        raise ValueError("开始时间必须不晚于结束时间")
    return start.ceil("min"), stop


def month_intervals(start, stop):
    current, stop = local_time(start), local_time(stop)
    while current < stop:
        end = min((current.to_period("M") + 1).start_time, stop)
        yield current, end
        current = end


def read_table(dai, table, columns, start, stop):
    for identifier in [table, *columns]:
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
            raise ValueError(f"非法 SQL 标识符: {identifier!r}")
    start, stop = local_time(start), local_time(stop)
    lo, hi = start.isoformat(sep=" "), stop.isoformat(sep=" ")
    frame = dai.query(
        f"SELECT {', '.join(columns)} FROM {table} WHERE date >= '{lo}' AND date < '{hi}'",
        filters={"date": [start.normalize().strftime("%Y-%m-%d"), stop.ceil("D").strftime("%Y-%m-%d")]},
    ).df()
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"数据表缺列: {sorted(missing)}")
    frame = frame[columns].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{table} 存在空键或重复键")
    return frame.sort_values(KEYS).reset_index(drop=True)


def load_universe(dai, start, stop, table="cpt_jyc_2026_instruments"):
    return read_table(dai, table, KEYS, start, stop)


def build_windows(bars, keys, config):
    """每个键只取 <= t 且在 history_days 内的最后 L 条常规交易 Bar。

    四个价格字段除以窗口内最后一个有效 close 再减一；volume/amount
    分别除以该窗口内的有效均值再减一。这些统计只依赖 t 时已知的窗口。
    缺失/不足历史补归一化后的 0；无历史也输出同一模型的零输入预测。
    """
    spec = config["data"]
    fields, length = spec["fields"], int(spec["window_bars"])
    if fields != ["open", "high", "low", "close", "volume", "amount"]:
        raise ValueError("v1 的字段及顺序固定，修改处理方式需显式更新实现")
    keys = keys.reset_index(drop=True)
    if keys.duplicated(KEYS).any() or keys[KEYS].isna().any().any():
        raise ValueError("样本键重复或为空")
    bars = bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])
    if bars.duplicated(KEYS).any():
        raise ValueError("Bar 键重复")
    if spec["exclude_auction_bars"]:
        minute = bars["date"].dt.hour * 60 + bars["date"].dt.minute
        regular = ((minute > 570) & (minute <= 690)) | ((minute > 780) & (minute <= 900))
        bars = bars.loc[regular]
    bars = bars.sort_values(["instrument", "date"]).reset_index(drop=True)
    values = bars[fields].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    prices = values[:, :4]
    prices[prices <= 0] = np.nan
    quantities = values[:, 4:]
    quantities[quantities < 0] = np.nan
    times = bars["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    key_times = pd.to_datetime(keys["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    groups = bars.groupby("instrument", observed=True, sort=False).indices
    raw = np.full((len(keys), length, len(fields)), np.nan, dtype=np.float32)
    counts = np.zeros(len(keys), dtype=np.int16)
    max_age = pd.Timedelta(days=spec["history_days"]).value
    offsets = np.arange(length - 1, -1, -1)
    for instrument, positions in keys.groupby("instrument", observed=True, sort=False).indices.items():
        bar_positions = groups.get(instrument)
        if bar_positions is None or len(bar_positions) == 0:
            continue
        instrument_times = times[bar_positions]
        requested = key_times[positions]
        right = np.searchsorted(instrument_times, requested, side="right") - 1
        indices = right[:, None] - offsets
        safe = np.clip(indices, 0, len(bar_positions) - 1)
        selected_times = instrument_times[safe]
        valid = (indices >= 0) & (selected_times <= requested[:, None]) & (selected_times >= requested[:, None] - max_age)
        selected = values[bar_positions[safe]].copy()
        selected[~valid] = np.nan
        raw[positions] = selected
        counts[positions] = valid.sum(axis=1)
    close = raw[:, :, 3]
    last_close = np.where(np.isfinite(close), np.arange(length), -1).max(axis=1)
    anchors = close[np.arange(len(raw)), np.maximum(last_close, 0)]
    anchors[last_close < 0] = np.nan
    raw[:, :, :4] = raw[:, :, :4] / anchors[:, None, None] - 1.0
    for column in (4, 5):
        quantity = raw[:, :, column]
        valid = np.isfinite(quantity)
        mean = np.nansum(quantity, axis=1, dtype=np.float64) / np.maximum(valid.sum(axis=1), 1)
        mean[mean <= 0] = 1.0
        raw[:, :, column] = quantity / mean[:, None] - 1.0
    np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(raw, -spec["clip"], spec["clip"], out=raw)
    quality = {
        "samples": len(keys),
        "no_history_samples": int((counts == 0).sum()),
        "short_history_samples": int((counts < length).sum()),
    }
    return raw.reshape(len(keys), -1), quality


def load_month(dai, datasources, start, stop, config, protocol=None):
    source = datasources[config["data"]["logical_source"]]
    universe_table = protocol["universe_table"] if protocol else "cpt_jyc_2026_instruments"
    keys = load_universe(dai, start, stop, universe_table)
    history = local_time(start) - pd.Timedelta(days=config["data"]["history_days"])
    bars = read_table(dai, source, KEYS + config["data"]["fields"], history, stop)
    x, quality = build_windows(bars, keys, config)
    if protocol is not None:
        labels = read_table(dai, protocol["label_table"], KEYS + [protocol["label_column"]], start, stop)
        keys = keys.merge(labels, on=KEYS, how="left", validate="one_to_one")
        label_end = keys["date"] + pd.Timedelta(minutes=protocol["label_horizon_minutes"])
        allowed = (label_end < local_time(stop)) & (label_end.dt.normalize() == keys["date"].dt.normalize())
        keys.loc[~allowed, protocol["label_column"]] = np.nan
    return x, keys, quality
