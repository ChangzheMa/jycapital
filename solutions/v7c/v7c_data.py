"""四频因果窗口；组内全部 z/c 有效才置 1，保留组内其余有效数值。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v7c_preprocessing import output_columns
from v7c_schema import (INPUT_CHANNELS, NON_BOOK_VALUE_INDICES, BOOK_VALUE_INDICES,
                       GROUP_MASK_SLICE, BAR_CLOCK_SLICE, LOG_AGE_INDEX,
                       validate_input_config)

KEYS = ["date", "instrument"]
FREQUENCIES = ("bar1m", "bar5m", "bar15m", "bar30m")
MINUTE_NS = 60_000_000_000
VALUE_COLUMNS = output_columns()[2:]


def regular_times(days, frequency):
    step = int(frequency[3:-1])
    minutes = np.r_[np.arange(570 + step, 691, step), np.arange(780 + step, 901, step)]
    days = np.asarray(days, dtype="datetime64[ns]").astype(np.int64)
    return (days[:, None] + minutes[None, :] * MINUTE_NS).reshape(-1)


def clock_values(times):
    minutes = (np.asarray(times, dtype=np.int64) // MINUTE_NS) % 1440
    phase = 2 * np.pi * minutes / 1440
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)


class WindowBlock:
    def __init__(self, keys, grids, stocks, config):
        validate_input_config(config)
        self.keys = keys.reset_index(drop=True)
        if self.keys[KEYS].isna().any().any() or self.keys.duplicated(KEYS).any():
            raise ValueError("样本键重复或为空")
        self.config, self.grids = config, grids
        self.times = self.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        self.stock_indices = pd.Index(stocks).get_indexer(self.keys.instrument)
        if (self.stock_indices < 0).any():
            raise ValueError("网格缺少目标股票")
        self.right = {f: np.searchsorted(g[0], self.times, side="right") - 1 for f, g in grids.items()}

    def batch(self, indices):
        indices = np.asarray(indices)
        requested, stocks = self.times[indices], self.stock_indices[indices]
        windows = {}
        max_age = self.config["data"]["history_days"] * 1440
        for frequency, (times, values) in self.grids.items():
            length = self.config["data"]["window_bars"][frequency]
            positions = self.right[frequency][indices, None] - np.arange(length - 1, -1, -1)
            if not len(times):
                windows[frequency] = np.zeros((len(indices), INPUT_CHANNELS, length), np.float32)
                continue
            safe = positions.clip(0, len(times) - 1)
            age = (requested[:, None] - times[safe]) / MINUTE_NS
            eligible = (positions >= 0) & (age >= 0) & (age <= max_age)
            raw = values[safe, stocks[:, None]]
            mask = np.isfinite(raw) & eligible[..., None]
            x = np.zeros((*raw.shape[:2], INPUT_CHANNELS), np.float32)
            x[:, :, :72] = np.where(mask, raw, 0)
            x[:, :, GROUP_MASK_SLICE] = np.stack([
                mask[:, :, NON_BOOK_VALUE_INDICES].all(axis=2),
                mask[:, :, BOOK_VALUE_INDICES].all(axis=2),
            ], axis=-1)
            # 两组都不完整仍可能有有效数值；整根 Bar 的可用性沿用逐值 any。
            bar_valid = mask.any(axis=2)
            x[:, :, BAR_CLOCK_SLICE] = clock_values(times[safe]) * bar_valid[..., None]
            x[:, :, LOG_AGE_INDEX] = (np.log1p(age.clip(0, max_age)) / np.log1p(max_age)) * bar_valid
            windows[frequency] = np.ascontiguousarray(x.transpose(0, 2, 1))
        return windows, clock_values(requested)


def section_indices(keys, seed=None):
    """按完整时间戳分组；只打乱截面顺序，不截断或重复股票。"""
    ordered = keys.assign(_position=np.arange(len(keys))).sort_values(KEYS)
    groups = [part._position.to_numpy(dtype=np.int64)
              for _, part in ordered.groupby("date", sort=True)]
    if seed is not None:
        order = np.random.default_rng(seed).permutation(len(groups))
        groups = [groups[i] for i in order]
    return groups


def structural_reason(keys, windows, clock, label_column, min_count):
    labels = keys[label_column].to_numpy(dtype=np.float64)
    eligible = np.isfinite(labels)
    count = int(eligible.sum())
    if count < min_count:
        return "insufficient_labels", count
    if np.ptp(labels[eligible]) == 0:
        return "constant_labels", count
    if all(np.equal(x[eligible], x[eligible][0]).all() for x in windows.values()) and \
            np.equal(clock[eligible], clock[eligible][0]).all():
        return "identical_inputs", count
    return None, count


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
        interval = self.protocol["splits"][split]
        lo, hi = max(lo, pd.Timestamp(interval["start"])), min(hi, pd.Timestamp(interval["stop"]))
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
        history_start = lo - pd.Timedelta(days=self.config["data"]["history_days"])
        for frequency in FREQUENCIES:
            all_times = regular_times(self.days[(self.days >= history_start.to_datetime64()) & (self.days < hi.to_datetime64())], frequency)
            left = np.searchsorted(all_times, lo.value)
            length = self.config["data"]["window_bars"][frequency]
            times = all_times[max(0, left - length):]
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
        return WindowBlock(keys, grids, stocks, self.config)
