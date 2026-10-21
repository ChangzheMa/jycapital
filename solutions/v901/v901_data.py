"""三路固定交易日窗口；未来位置补零，保留组完整和 Bar 可用标志。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v901_preprocessing import output_columns
from v901_schema import (INPUT_CHANNELS, NON_BOOK_VALUE_INDICES, BOOK_VALUE_INDICES,
                       GROUP_MASK_SLICE, BAR_HAS_VALUE_INDEX, FREQUENCIES,
                       TRADING_DAY_WINDOWS, validate_input_config, PREDICTION_TIMES)

KEYS = ["date", "instrument"]
MINUTE_NS = 60_000_000_000
VALUE_COLUMNS = output_columns()[2:]


def minute_slots(frequency):
    step = int(frequency[3:-1])
    return np.r_[np.arange(570 + step, 691, step), np.arange(780 + step, 901, step)]


def regular_times(days, frequency):
    minutes = minute_slots(frequency)
    days = np.asarray(days, dtype="datetime64[ns]").astype(np.int64)
    return (days[:, None] + minutes[None, :] * MINUTE_NS).reshape(-1)


class TradingCalendar:
    """使用股票池的全市场交易日期；个股停牌或缺 Bar 不改变位置。"""

    def __init__(self, days):
        dates = pd.DatetimeIndex(pd.to_datetime(days))
        if dates.tz is not None:
            dates = dates.tz_convert("Asia/Shanghai").tz_localize(None)
        if dates.hasnans:
            raise ValueError("交易日历不能含 NaT")
        self.days = np.unique(dates.normalize().to_numpy(dtype="datetime64[ns]").astype(np.int64))

    def history_start(self, start, previous_days):
        start = pd.Timestamp(start).normalize()
        index = int(np.searchsorted(self.days, start.value))
        return pd.Timestamp(self.days[max(0, index - previous_days)]) if index else start

    def window_slots(self, days, frequency):
        days = np.asarray(days, dtype=np.int64)
        indices = np.searchsorted(self.days, days)
        if np.any(indices >= len(self.days)) or not np.array_equal(self.days[indices], days):
            raise ValueError("预测日期不在全市场交易日历中")
        spec = TRADING_DAY_WINDOWS[frequency]
        offsets = np.arange(-spec["previous_days"], int(spec["include_today"]))
        selected = indices[:, None] + offsets[None, :]
        available = selected >= 0
        dates = self.days[selected.clip(0)]
        minutes = minute_slots(frequency)
        stamps = dates[:, :, None] + minutes[None, None, :] * MINUTE_NS
        # 缺少历史交易日时保留左侧整日零槽位，不回退成更短或错位窗口。
        stamps = np.where(available[:, :, None], stamps, 0)
        return stamps.reshape(len(days), -1)


def prediction_clock_values(times):
    """仅编码 UTC+08:00 本地预测时刻；Bar 不再携带时间通道。"""
    stamps = np.asarray(times, dtype=np.int64)
    minute = (stamps // MINUTE_NS) % 1440
    slots = np.array([int(value[:2]) * 60 + int(value[3:]) for value in PREDICTION_TIMES])
    matches = minute[..., None] == slots
    if np.any(stamps == np.iinfo(np.int64).min) or np.any(stamps % MINUTE_NS != 0) or \
            not np.all(matches.sum(axis=-1) == 1):
        raise ValueError("预测时刻不属于固定八类，或未对齐到整分钟")
    return matches.astype(np.float32)


class WindowBlock:
    def __init__(self, keys, grids, stocks, config, trading_days):
        validate_input_config(config)
        self.keys = keys.reset_index(drop=True)
        if self.keys[KEYS].isna().any().any() or self.keys.duplicated(KEYS).any():
            raise ValueError("样本键重复或为空")
        if set(grids) != set(FREQUENCIES):
            raise ValueError("输入网格必须恰好包含 1、5、30 分钟三路")
        self.config, self.grids = config, grids
        self.times = self.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        prediction_clock_values(self.times)
        self.stock_indices = pd.Index(stocks).get_indexer(self.keys.instrument)
        if (self.stock_indices < 0).any():
            raise ValueError("网格缺少目标股票")
        calendar = TradingCalendar(trading_days)
        days = self.keys.date.to_numpy(dtype="datetime64[D]").astype("datetime64[ns]").astype(np.int64)
        unique_days, self.day_indices = np.unique(days, return_inverse=True)
        self.plans = {}
        for frequency, (times, values) in grids.items():
            if values.shape != (len(times), len(stocks), 72) or np.any(np.diff(times) <= 0):
                raise ValueError("规则网格形状错误或时间重复／未排序")
            stamps = calendar.window_slots(unique_days, frequency)
            positions = np.searchsorted(times, stamps)
            if len(times):
                matched = (positions < len(times)) & (stamps != 0)
                matched &= times[positions.clip(0, len(times) - 1)] == stamps
                positions = np.where(matched, positions, -1)
            else:
                positions.fill(-1)
            self.plans[frequency] = stamps, positions

    def batch(self, indices):
        indices = np.asarray(indices)
        requested, stocks = self.times[indices], self.stock_indices[indices]
        windows = {}
        for frequency, (times, values) in self.grids.items():
            length = self.config["data"]["window_bars"][frequency]
            stamps, positions = (item[self.day_indices[indices]] for item in self.plans[frequency])
            if not len(times):
                windows[frequency] = np.zeros((len(indices), INPUT_CHANNELS, length), np.float32)
                continue
            safe = positions.clip(0)
            eligible = (positions >= 0) & (stamps <= requested[:, None])
            raw = values[safe, stocks[:, None]]
            if self.config["data"].get("input_precision", "float32") == "fp16_with_fp32_overflow":
                with np.errstate(over="ignore", invalid="ignore"):
                    rounded = raw.astype(np.float16).astype(np.float32)
                overflow = np.isfinite(raw) & ~np.isfinite(rounded)
                raw = np.where(overflow, raw, rounded)
            mask = np.isfinite(raw) & eligible[..., None]
            x = np.zeros((*raw.shape[:2], INPUT_CHANNELS), np.float32)
            x[:, :, :72] = np.where(mask, raw, 0)
            x[:, :, GROUP_MASK_SLICE] = np.stack([
                mask[:, :, NON_BOOK_VALUE_INDICES].all(axis=2),
                mask[:, :, BOOK_VALUE_INDICES].all(axis=2),
            ], axis=-1)
            x[:, :, BAR_HAS_VALUE_INDEX] = mask.any(axis=2)
            windows[frequency] = np.ascontiguousarray(x.transpose(0, 2, 1))
        return windows, prediction_clock_values(requested)


def section_indices(keys, seed=None):
    """按完整时间戳分组；只打乱截面顺序，不截断或重复股票。"""
    ordered = keys.assign(_position=np.arange(len(keys))).sort_values(KEYS)
    groups = [part._position.to_numpy(dtype=np.int64)
              for _, part in ordered.groupby("date", sort=True)]
    if seed is not None:
        order = np.random.default_rng(seed).permutation(len(groups))
        groups = [groups[i] for i in order]
    return groups


class ProcessedStore:
    def __init__(self, training_run, holdout_run, data_root, config, protocol):
        self.config, self.protocol = config, protocol
        self.training_run, self.holdout_run = Path(training_run), Path(holdout_run)
        self.data_root = Path(data_root)
        self.universe = pd.read_parquet(self.data_root / (protocol["universe_table"] + ".parquet"), columns=KEYS)
        self.universe["date"] = pd.to_datetime(self.universe.date)
        self.days = np.sort(self.universe.date.dt.normalize().unique())

    def months(self, split):
        interval = self.protocol["splits"][split]
        return [str(m) for m in pd.period_range(interval["start"], pd.Timestamp(interval["stop"]) - pd.Timedelta(1, "ns"), freq="M")]

    def path(self, frequency, month):
        start = pd.Period(month, "M").start_time
        for split in ("train", "validation", "test"):
            interval = self.protocol["splits"][split]
            if pd.Timestamp(interval["start"]) <= start < pd.Timestamp(interval["stop"]):
                directory = self.training_run / "data" if split == "train" else self.holdout_run / "data" / split
                path = directory / frequency / f"{month}.parquet"
                if not path.is_file():
                    raise FileNotFoundError(path)
                return path
        return None

    def load(self, split, month):
        lo = pd.Period(month, "M").start_time
        hi = (lo.to_period("M") + 1).start_time
        return self.load_range(split, lo, hi)

    def load_range(self, split, lo, hi):
        """按有限日期范围加载，供全量常驻模式的验证和测试使用。"""
        interval = self.protocol["splits"][split]
        lo, hi = max(pd.Timestamp(lo), pd.Timestamp(interval["start"])), min(pd.Timestamp(hi), pd.Timestamp(interval["stop"]))
        if lo >= hi:
            raise ValueError("评估日期范围为空")
        keys = self.universe.loc[(self.universe.date >= lo) & (self.universe.date < hi), KEYS].sort_values(KEYS).reset_index(drop=True)
        labels = pq.read_table(self.data_root / (self.protocol["label_table"] + ".parquet"),
                               columns=KEYS + [self.protocol["label_column"]],
                               filters=[("date", ">=", lo), ("date", "<", hi)]).to_pandas()
        keys = keys.merge(labels, on=KEYS, how="left", validate="one_to_one")
        horizon = keys.date + pd.Timedelta(minutes=self.protocol["label_horizon_minutes"])
        admissible = (horizon < pd.Timestamp(interval["stop"])) & (horizon.dt.normalize() == keys.date.dt.normalize())
        keys.loc[~admissible, self.protocol["label_column"]] = np.nan
        # 目标股票加上当前/前缀股票无需产生池外训练样本；只存本月目标股票的历史。
        stocks = pd.Index(sorted(keys.instrument.unique()))
        grids = {}
        calendar = TradingCalendar(self.days)
        for frequency in FREQUENCIES:
            history_start = calendar.history_start(lo, TRADING_DAY_WINDOWS[frequency]["previous_days"])
            times = regular_times(self.days[(self.days >= history_start.to_datetime64()) & (self.days < hi.to_datetime64())], frequency)
            values = np.full((len(times), len(stocks), 72), np.nan, dtype=np.float32)
            if len(times):
                first = pd.Timestamp(times[0])
                for period in pd.period_range(first, hi - pd.Timedelta(1, "ns"), freq="M"):
                    path = self.path(frequency, str(period))
                    if path is None:
                        continue
                    table = pq.read_table(path, columns=KEYS + VALUE_COLUMNS,
                                          filters=[("date", ">=", first), ("date", "<", hi)])
                    frame = table.to_pandas()
                    del table
                    raw_times = frame.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                    t = pd.Index(times).get_indexer(raw_times)
                    s = stocks.get_indexer(frame.instrument)
                    take = (t >= 0) & (s >= 0)
                    values[t[take], s[take]] = frame.loc[take, VALUE_COLUMNS].to_numpy(dtype=np.float32)
                    del frame
            grids[frequency] = times, values
        return WindowBlock(keys, grids, stocks, self.config, self.days)
