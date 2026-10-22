"""平台历史按月取数、按日保留数组，避免反复拼接一整年网格。"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from long_history import history_positions, prepare_values


class MonthlyRawHistory:
    def __init__(self, read, dai, source, columns, start, stop):
        self.read, self.dai, self.source, self.columns = read, dai, source, columns
        self.start, self.stop = pd.Timestamp(start), pd.Timestamp(stop)
        self.month, self.frame = None, None

    def day(self, lo, hi):
        month = lo.to_period("M")
        if self.month != month:
            # 可提前取后续日期，但仅把当前日期切片交给顺序归一化。
            self.frame = None
            start = max(month.start_time, self.start)
            stop = min((month + 1).start_time, self.stop)
            self.frame = self.read(self.dai, self.source, self.columns, start, stop)
            self.month = month
        return self.frame.loc[(self.frame.date >= lo) & (self.frame.date < hi)].reset_index(drop=True)


class DailyLongHistory:
    def __init__(self, stocks, calendar, config):
        self.stocks = pd.Index(stocks)
        self.calendar = np.asarray(calendar, dtype="datetime64[ns]")
        self.horizon = config["long_context"]["previous_days"]
        self.precision = config["data"]["input_precision"]
        self.frames = OrderedDict()

    def append(self, day, values):
        stamp = pd.Timestamp(day).value
        if values.shape != (8, len(self.stocks), 72):
            raise ValueError("平台长期历史要求完整八个规则槽位")
        if self.frames and stamp <= next(reversed(self.frames)):
            raise ValueError("平台历史必须逐日推进，不覆盖已有日期")
        self.frames[stamp] = values
        # 当前日保留供后续请求日使用，本日取窗仍严格排除它。
        while len(self.frames) > self.horizon + 1:
            self.frames.popitem(last=False)

    def batch(self, keys):
        days, inverse = np.unique(keys.date.to_numpy(dtype="datetime64[D]"), return_inverse=True)
        positions = history_positions(self.calendar, days, self.horizon)
        stock = self.stocks.get_indexer(keys.instrument)
        if (stock < 0).any():
            raise ValueError("平台长期历史缺少请求股票")
        raw = np.full((len(keys), self.horizon, 8, 72), np.nan, np.float32)
        for group, indices in enumerate(positions):
            rows = np.flatnonzero(inverse == group)
            for offset, day in enumerate(indices):
                if day < 0:
                    continue
                stamp = int(self.calendar[day].astype(np.int64))
                values = self.frames.get(stamp)
                if values is None:
                    raise ValueError("所需历史日未读取或已被错误淘汰")
                raw[rows, offset] = values[:, stock[rows], :].transpose(1, 0, 2)
        return prepare_values(raw, self.precision)
