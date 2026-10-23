"""保留父几何输入，增补原始 FP32 对数价格中心及有效性。"""
import copy
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from price_geometry import (PRICE_FIELDS, CONTRACT as PARENT_CONTRACT,
                            NORMALIZATION as PARENT_NORMALIZATION, VALUE_COLUMNS as PARENT_COLUMNS,
                            transform_frame as parent_transform_frame)

KEYS = ['date', 'instrument']
FREQUENCIES = ('bar1m', 'bar5m', 'bar30m')
DAILY_BARS = {'bar1m': 240, 'bar5m': 48, 'bar30m': 8}
PREVIOUS_DAYS = {'bar1m': 1, 'bar5m': 5, 'bar30m': 20}
INPUT_CHANNELS = 77
STORAGE_CHANNELS = 73
STORAGE_COLUMNS = PARENT_COLUMNS + ['log_price_reference']
PRECISION = 'price_memory_fp16_with_fp32_anchor'
NORMALIZATION = 'frozen_parent_geometry_with_fp32_price_memory'
CONTRACT = {'version': 'causal_price_memory_v1', 'parent_view': 'unchanged_75_geometry_channels',
            'stored_reference': 'same_closed_bar_log_close_else_first_valid_price_fp32',
            'query_reference': 'latest_valid_closed_bar_at_query', 'distance': '100_log_anchor_difference',
            'network_layout': '75_parent_1_log_anchor_1_anchor_available', 'network_channels': INPUT_CHANNELS}


def validate_price_memory_config(config):
    data = config['data']
    if data.get('normalization') != NORMALIZATION or data.get('price_memory') != CONTRACT:
        raise ValueError('价格历史输入契约不一致')
    if data.get('price_geometry') != PARENT_CONTRACT or data.get('input_precision') not in (PRECISION, 'float32'):
        raise ValueError('父价格几何或混合精度契约不一致')


def base_configuration(config):
    result = copy.deepcopy(config)
    result['solution_id'] = 'v1601'
    result['model']['input_channels'] = 75
    result['data'].pop('price_memory', None)
    result['data']['normalization'] = PARENT_NORMALIZATION
    result['data']['input_precision'] = 'fp16_with_fp32_overflow'
    result['training']['resident_cache']['format'] = 'jycapital-resident-price-geometry-fp16-v1'
    return result


def base_data_manifest(manifest):
    return {key: value for key, value in manifest.items() if key != 'price_memory'}


def anchor_view(raw):
    """与已冻结行内几何和 v2001 FP32 中心使用同一价格选择顺序。"""
    prices = raw[PRICE_FIELDS].to_numpy(dtype=np.float64)
    valid = np.isfinite(prices) & (prices > 0)
    order = [PRICE_FIELDS.index(field) for field in PARENT_CONTRACT['fallback_order']]
    first = np.asarray(order)[valid[:, order].argmax(axis=1)] if len(prices) else np.empty(0, np.int64)
    anchor = prices[np.arange(len(prices)), first]
    anchor = np.where(valid.any(axis=1), anchor, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(anchor).astype(np.float32)


def prepare_price_memory_windows(raw, repairs, clock):
    result = {}
    for f in FREQUENCIES:
        # 与冻结父准备函数逐值一致，纯数值逻辑不引入训练缓存或psutil依赖。
        values = raw[f][0].float()
        indices, exceptional = repairs[f]
        if indices.numel():
            values.reshape(-1).index_copy_(0, indices, exceptional)
        cutoff = PREVIOUS_DAYS[f] * DAILY_BARS[f] + clock.argmax(dim=1) * (DAILY_BARS[f] // 8)
        visible = torch.arange(values.shape[1], device=values.device)[None, :] < cutoff[:, None]
        finite = torch.isfinite(values) & visible[:, :, None]
        non_book = torch.cat((finite[:, :, :6], finite[:, :, 36:42]), dim=2).all(dim=2)
        book = torch.cat((finite[:, :, 6:36], finite[:, :, 42:72]), dim=2).all(dim=2)
        flags = torch.stack((non_book, book, finite.any(dim=2)), dim=2).float()
        parent = torch.cat((torch.where(finite, values, 0), flags), dim=2).transpose(1, 2)
        anchor = raw[f][1].float()
        valid = torch.isfinite(anchor) & (parent[:, 74, :] > 0)
        reference = torch.where(valid, anchor, torch.zeros_like(anchor))
        result[f] = torch.cat((parent, reference[:, None, :], valid[:, None, :].float()), dim=1).contiguous()
    return result


def transform_frame(normalized, raw):
    result = parent_transform_frame(normalized, raw)
    result['log_price_reference'] = anchor_view(raw)
    return result[KEYS + STORAGE_COLUMNS]


def raw_batches(data_root, frequency, months, start, stop, *, stocks=None, batch_rows=65536):
    """仅读取授权原始价格；不读取数量、收益标签或 exposure。"""
    columns = KEYS + PRICE_FIELDS
    for month in months:
        path = Path(data_root) / ('cpt_jyc_2026_stock_' + frequency) / (str(month) + '.parquet')
        if not path.is_file():
            raise FileNotFoundError(path)
        predicate = (ds.field('date') >= pd.Timestamp(start).to_pydatetime()) & (ds.field('date') < pd.Timestamp(stop).to_pydatetime())
        if stocks is not None:
            predicate = predicate & ds.field('instrument').isin(list(stocks))
        scanner = ds.dataset(path, format='parquet').scanner(columns=columns, filter=predicate,
                                                           batch_size=batch_rows, use_threads=False)
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            frame['date'] = pd.to_datetime(frame.date)
            if frame[KEYS].isna().any().any():
                raise ValueError('原始价格键含空值')
            yield frame
