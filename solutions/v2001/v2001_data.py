"""复用冻结基础输入，增补数量共同坐标与 FP32 价格参考。"""
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from artifacts import read_json, sha256
from coordinate_inputs import (KEYS, FREQUENCIES, STORAGE_CHANNELS, PRECISION, coordinate_parameters,
    STORAGE_COLUMNS as VALUE_COLUMNS, base_configuration, validate_coordinate_config,
    raw_batches, transform_side, prepare_coordinate_windows)
from v1601_data import (ProcessedStore as BaseProcessedStore, WindowBlock as BaseWindowBlock,
    TradingCalendar, regular_times, minute_slots, prediction_clock_values, section_indices)


class WindowBlock(BaseWindowBlock):
    def __init__(self, keys, grids, stocks, config, trading_days, parameters):
        validate_coordinate_config(config)
        for times, values in grids.values():
            if values.shape != (len(times), len(stocks), STORAGE_CHANNELS):
                raise ValueError("坐标网格要求 73 个存储通道")
        super().__init__(keys, {f: (times, values[..., :72]) for f, (times, values) in grids.items()},
                         stocks, base_configuration(config), trading_days)
        self.config, self.coordinate_grids, self.parameters = config, grids, parameters

    def batch(self, indices):
        indices = np.asarray(indices)
        requested, stocks = self.times[indices], self.stock_indices[indices]
        clock = prediction_clock_values(requested)
        raw, repairs = {}, {}
        for frequency, (times, values) in self.coordinate_grids.items():
            stamps, positions = (item[self.day_indices[indices]] for item in self.plans[frequency])
            eligible = (positions >= 0) & (stamps <= requested[:, None])
            if len(times):
                selected = values[positions.clip(0), stocks[:, None]].copy()
                selected[~eligible] = np.nan
            else:
                selected = np.full((len(indices), self.config["data"]["window_bars"][frequency], 73), np.nan, np.float32)
            codes = selected[..., :72].copy()
            positions_repair, values_repair = np.empty(0, np.int64), np.empty(0, np.float32)
            if self.config["data"]["input_precision"] == PRECISION:
                with np.errstate(over="ignore", invalid="ignore"):
                    half = codes.astype(np.float16)
                overflow = np.isfinite(codes) & ~np.isfinite(half)
                positions_repair = np.flatnonzero(overflow).astype(np.int64)
                values_repair = codes.reshape(-1)[positions_repair].copy()
                codes = half
            raw[frequency] = (torch.from_numpy(codes), torch.from_numpy(selected[..., 72].copy()))
            repairs[frequency] = (torch.from_numpy(positions_repair), torch.from_numpy(values_repair))
        windows = prepare_coordinate_windows(raw, repairs, torch.from_numpy(clock), self.parameters)
        return {f: values.numpy() for f, values in windows.items()}, clock


class ProcessedStore:
    def __init__(self, training_run, holdout_run, data_root, config, protocol):
        validate_coordinate_config(config)
        self.base = BaseProcessedStore(training_run, holdout_run, data_root, base_configuration(config), protocol)
        self.__dict__.update({name: value for name, value in self.base.__dict__.items() if name != "config"})
        self.config = config
        self.parameters, self.statistics_hashes = {}, {}
        for frequency in FREQUENCIES:
            path = Path(training_run) / "normalization" / (frequency + ".json")
            statistics = read_json(path)
            for key in ("start", "stop"):
                if pd.Timestamp(statistics["fit_interval"][key]) != pd.Timestamp(protocol["splits"]["train"][key]):
                    raise ValueError("共同坐标统计不对应固定训练区间")
            self.parameters[frequency] = coordinate_parameters(statistics)
            self.statistics_hashes[frequency] = sha256(path)

    def months(self, split):
        return self.base.months(split)

    def path(self, frequency, month):
        return self.base.path(frequency, month)

    def load(self, split, month):
        lo = pd.Period(month, "M").start_time
        return self.load_range(split, lo, (lo.to_period("M") + 1).start_time)

    def load_range(self, split, lo, hi):
        original = self.base.load_range(split, lo, hi)
        stocks = pd.Index(sorted(original.keys.instrument.unique()))
        grids = {}
        for frequency, (times, values) in original.grids.items():
            combined = np.full((*values.shape[:2], 73), np.nan, np.float32)
            combined[..., :52] = values[..., :52]
            if len(times):
                start, stop = pd.Timestamp(times[0]), pd.Timestamp(times[-1]) + pd.Timedelta(minutes=1)
                months = [str(month) for month in pd.period_range(start, stop - pd.Timedelta(1, "ns"), freq="M")
                          if self.path(frequency, str(month)) is not None]
                seen = np.zeros(values.shape[:2], bool)
                for raw in raw_batches(self.data_root, frequency, months, start, stop, stocks=stocks):
                    ti = pd.Index(times).get_indexer(raw.date.to_numpy(dtype="datetime64[ns]").astype(np.int64))
                    si = stocks.get_indexer(raw.instrument.astype(str))
                    selected = (ti >= 0) & (si >= 0)
                    ti, si = ti[selected], si[selected]
                    codes = ti.astype(np.int64) * len(stocks) + si
                    if len(np.unique(codes)) != len(codes) or seen[ti, si].any():
                        raise ValueError("原始坐标存在重复网格位置")
                    quantity, anchor = transform_side(raw.loc[selected], self.parameters[frequency])
                    combined[ti, si, 52:72] = quantity
                    combined[ti, si, 72] = anchor
                    seen[ti, si] = True
                if (np.isfinite(values).any(axis=-1) & ~seen).any():
                    raise ValueError("原始坐标缺少基础预处理的已有记录")
            grids[frequency] = times, combined
        return WindowBlock(original.keys, grids, stocks, self.config, self.days, self.parameters)
