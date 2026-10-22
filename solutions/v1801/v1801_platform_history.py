"""复用已归一化的1分钟行情，保存盘前槽位；不新增平台查询。"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from preopen_context import BARS, MINUTES, MINUTE_NS, VALUE_COLUMNS, history_positions, prepare_values


def preopen_grid(normalized, day, stocks):
    times = pd.Timestamp(day).value + MINUTES * MINUTE_NS
    values = np.full((BARS, len(stocks), 72), np.nan, np.float32)
    if not normalized.empty:
        ti = pd.Index(times).get_indexer(normalized.date.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        si = pd.Index(stocks).get_indexer(normalized.instrument)
        valid = (ti >= 0) & (si >= 0)
        values[ti[valid], si[valid]] = normalized.loc[valid, VALUE_COLUMNS].to_numpy(dtype=np.float32)
    return times, values


class DailyPreopenContext:
    def __init__(self, stocks, calendar, config):
        self.stocks = pd.Index(stocks)
        self.calendar = np.asarray(calendar, dtype="datetime64[ns]")
        self.horizon = config["preopen_context"]["previous_days"]
        self.precision = config["data"]["input_precision"]
        self.frames = OrderedDict()

    def append(self, day, values):
        stamp = pd.Timestamp(day).value
        if values.shape != (BARS, len(self.stocks), 72):
            raise ValueError("平台盘前历史要求固定十个槽位")
        if self.frames and stamp <= next(reversed(self.frames)):
            raise ValueError("平台盘前历史必须逐日推进")
        self.frames[stamp] = values
        while len(self.frames) > self.horizon + 1:
            self.frames.popitem(last=False)

    def batch(self, keys):
        days, inverse = np.unique(keys.date.to_numpy(dtype="datetime64[D]"), return_inverse=True)
        positions = history_positions(self.calendar, days, self.horizon)
        stock = self.stocks.get_indexer(keys.instrument)
        if (stock < 0).any():
            raise ValueError("平台盘前历史缺少请求股票")
        raw = np.full((len(keys), self.horizon + 1, BARS, 72), np.nan, np.float32)
        for group, indices in enumerate(positions):
            rows = np.flatnonzero(inverse == group)
            for offset, day in enumerate(indices):
                if day < 0:
                    continue
                stamp = int(self.calendar[day].astype(np.int64))
                values = self.frames.get(stamp)
                if values is None:
                    raise ValueError("所需盘前日期未读取或已被错误淘汰")
                raw[rows, offset] = values[:, stock[rows], :].transpose(1, 0, 2)
        return prepare_values(raw, self.precision)
