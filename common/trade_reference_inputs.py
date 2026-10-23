"""完整父输入加成交笔数、前收盘参考和复权因子；固定变换、不读取收益标签。"""
import copy
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import price_memory_inputs as parent
from price_memory_inputs import KEYS, FREQUENCIES, DAILY_BARS, PREVIOUS_DAYS, PRICE_FIELDS, anchor_view

EXTRA_FIELDS = ['deal_number', 'pre_close', 'adjust_factor']
EXTRA_COLUMNS = ['log1p_deal_number', 'log_pre_close', 'log_adjust_factor']
STORAGE_COLUMNS = parent.STORAGE_COLUMNS + EXTRA_COLUMNS
STORAGE_CHANNELS = 76
INPUT_CHANNELS = 83
PRECISION = 'trade_reference_fp16_with_fp32_side'
NORMALIZATION = 'frozen_parent_geometry_with_price_memory_and_trade_reference'
CONTRACT = {'version':'closed_bar_trade_reference_v1', 'parent_view':'unchanged_77_price_memory_channels',
    'extra_fields':EXTRA_FIELDS, 'transforms':['log1p_nonnegative','log_positive','log_positive'],
    'fit_statistics':False, 'extra_storage_dtype':'float32', 'network_layout':'77_parent_3_values_3_validity',
    'network_channels':INPUT_CHANNELS, 'day_reference':'provided_pre_close_on_same_observed_bar',
    'adjustment':'provided_factor_as_input_only_no_price_rescaling'}


def validate_trade_reference_config(config):
    data = config['data']
    if data.get('normalization') != NORMALIZATION or data.get('trade_reference') != CONTRACT:
        raise ValueError('成交与日级参考输入契约不一致')
    if data.get('input_precision') not in (PRECISION, 'float32'):
        raise ValueError('成交与日级参考精度契约不一致')
    parent.validate_price_memory_config(parent_configuration(config))


def parent_configuration(config):
    result = copy.deepcopy(config)
    result['solution_id'] = 'v2401'
    result['model']['input_channels'] = 77
    result['data'].pop('trade_reference', None)
    result['data']['normalization'] = parent.NORMALIZATION
    result['data']['input_precision'] = 'float32' if config['data']['input_precision'] == 'float32' else parent.PRECISION
    result['training']['resident_cache']['format'] = 'jycapital-resident-price-memory-v1'
    return result


def base_configuration(config):
    return parent.base_configuration(parent_configuration(config))


def base_data_manifest(manifest):
    return parent.base_data_manifest({k:v for k,v in manifest.items() if k != 'trade_reference'})


def reference_view(raw):
    values = raw[EXTRA_FIELDS].to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    valid[:, 0] = np.isfinite(values[:, 0]) & (values[:, 0] >= 0)
    safe = np.where(valid, values, 1.)
    safe[:, 0] += 1.
    return np.where(valid, np.log(safe), np.nan).astype(np.float32)


def prepare_trade_reference_windows(raw, repairs, clock):
    result = parent.prepare_price_memory_windows({f:pair[:2] for f,pair in raw.items()}, repairs, clock)
    for f in FREQUENCIES:
        values = raw[f][2].float()
        cutoff = PREVIOUS_DAYS[f] * DAILY_BARS[f] + clock.argmax(dim=1) * (DAILY_BARS[f] // 8)
        visible = torch.arange(values.shape[1], device=values.device)[None, :] < cutoff[:, None]
        valid = torch.isfinite(values) & visible[:, :, None]
        # 新字段缺失不清除父字段，父字段缺失也不清除独立有效的新字段。
        extra = torch.cat((torch.where(valid, values, 0), valid.float()), dim=2).transpose(1, 2)
        result[f] = torch.cat((result[f], extra), dim=1).contiguous()
    return result


def transform_frame(normalized, raw):
    result = parent.transform_frame(normalized, raw)
    result[EXTRA_COLUMNS] = reference_view(raw)
    return result[KEYS + STORAGE_COLUMNS]


def raw_batches(data_root, frequency, months, start, stop, *, stocks=None, batch_rows=65536):
    columns = KEYS + PRICE_FIELDS + EXTRA_FIELDS
    for month in months:
        path = Path(data_root) / ('cpt_jyc_2026_stock_' + frequency) / (str(month) + '.parquet')
        if not path.is_file():
            raise FileNotFoundError(path)
        predicate = (ds.field('date') >= pd.Timestamp(start).to_pydatetime()) & (ds.field('date') < pd.Timestamp(stop).to_pydatetime())
        if stocks is not None:
            predicate = predicate & ds.field('instrument').isin(list(stocks))
        for batch in ds.dataset(path, format='parquet').scanner(columns=columns, filter=predicate,
                batch_size=batch_rows, use_threads=False).to_batches():
            frame = batch.to_pandas()
            frame['date'] = pd.to_datetime(frame.date)
            if frame[KEYS].isna().any().any():
                raise ValueError('原始行情键含空值')
            yield frame
