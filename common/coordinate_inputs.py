"""盘口共同数量坐标、FP32 价格参考及逐字段可用性的输入契约。"""
import copy
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from price_geometry import (FIELDS, PRICE_FIELDS, VALUE_COLUMNS as BASE_COLUMNS,
    CONTRACT as PRICE_CONTRACT, NORMALIZATION as BASE_NORMALIZATION, transform_frame as base_transform_frame)

KEYS = ["date", "instrument"]
FREQUENCIES = ("bar1m", "bar5m", "bar30m")
DAILY_BARS = {"bar1m": 240, "bar5m": 48, "bar30m": 8}
PREVIOUS_DAYS = {"bar1m": 1, "bar5m": 5, "bar30m": 20}
QUANTITY_FIELDS = [field for field in FIELDS if field.startswith(("ask_volume", "bid_volume", "ask_num_orders", "bid_num_orders"))]
QUANTITY_COLUMNS = [36 + FIELDS.index(field) for field in QUANTITY_FIELDS]
RETAINED_COLUMNS = [index for index in range(72) if index not in QUANTITY_COLUMNS]
assert RETAINED_COLUMNS == list(range(52)) and QUANTITY_COLUMNS == list(range(52, 72))
STORAGE_CHANNELS = 73
INPUT_CHANNELS = 112
STORAGE_COLUMNS = BASE_COLUMNS[:52] + ["q_" + field for field in QUANTITY_FIELDS] + ["log_price_reference"]
PRECISION = "coordinate_fp16_with_fp32_reference"
NORMALIZATION = "frozen_z_shared_quantity_geometry_and_fp32_price_reference"
CONTRACT = {"version": "causal_shared_quantity_price_reference_v1",
    "quantity_fields": QUANTITY_FIELDS, "quantity_coordinate": "log1p_with_pooled_training_global_group_moments",
    "quantity_groups": ["book_volume", "book_order_count"],
    "price_geometry": "unchanged_closed_bar_shared_price_center",
    "price_reference": "closed_bar_anchor_log_with_training_global_close_statistics",
    "price_reference_precision": "float32", "field_availability": "either_normalized_view_available",
    "channel_layout": "72_values_3_group_flags_36_field_flags_1_price_reference",
    "network_channels": INPUT_CHANNELS, "retained_parent_fp16_channels": len(RETAINED_COLUMNS)}


def validate_coordinate_config(config):
    data = config["data"]
    if data.get("normalization") != NORMALIZATION or data.get("coordinate_view") != CONTRACT:
        raise ValueError("数量坐标输入契约不一致")
    if data.get("price_geometry") != PRICE_CONTRACT or data.get("input_precision") not in (PRECISION, "float32"):
        raise ValueError("基础价格几何或输入精度不一致")


def base_configuration(config):
    result = copy.deepcopy(config)
    result["solution_id"] = "v1601"
    result["model"]["input_channels"] = 75
    result["data"].pop("coordinate_view", None)
    result["data"]["normalization"] = BASE_NORMALIZATION
    result["data"]["input_precision"] = "fp16_with_fp32_overflow"
    result["training"]["resident_cache"]["format"] = "jycapital-resident-price-geometry-fp16-v1"
    return result


def base_data_manifest(manifest):
    return {key: value for key, value in manifest.items() if key != "coordinate_view"}


def group_moments(statistics, fields):
    indices = [statistics["fields"].index(field) for field in fields]
    source = statistics["global"]
    count = np.asarray(source["count"], np.float64)[indices]
    mean = np.asarray(source["mean"], np.float64)[indices]
    m2 = np.asarray(source["m2"], np.float64)[indices]
    if (count < 0).any() or (m2 < 0).any() or not all(np.isfinite(value).all() for value in (count, mean, m2)):
        raise ValueError("训练统计包含非法计数或离差平方和")
    total = float(count.sum())
    if total <= 0:
        raise ValueError("训练期全局统计不足")
    center = float(np.dot(count, mean) / total)
    variance = float((m2 + count * (mean - center) ** 2).sum() / total)
    scale = float(np.sqrt(variance)) if variance > 1e-16 else 1.
    if not np.isfinite([center, scale]).all():
        raise ValueError("归一化参数非有限")
    return {"center": center, "scale": scale, "observations": int(total)}


def coordinate_parameters(statistics):
    return {"book_volume": group_moments(statistics, QUANTITY_FIELDS[:10]),
            "book_order_count": group_moments(statistics, QUANTITY_FIELDS[10:]),
            "price_anchor": group_moments(statistics, ["close"])}


def parameter_digest(parameters):
    return hashlib.sha256(json.dumps(parameters, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def transform_side(raw, parameters):
    """只变换原始数量和已闭合 Bar 的价格中心；没有标签参数。"""
    quantities = raw[QUANTITY_FIELDS].to_numpy(dtype=np.float64)
    valid = np.isfinite(quantities) & (quantities >= 0)
    values = np.log1p(np.where(valid, quantities, 0))
    for name, columns in (("book_volume", slice(0, 10)), ("book_order_count", slice(10, 20))):
        spec = parameters[name]
        values[:, columns] = (values[:, columns] - spec["center"]) / spec["scale"]
    values = np.where(valid, values, np.nan).astype(np.float32)
    prices = raw[PRICE_FIELDS].to_numpy(dtype=np.float64)
    valid_price = np.isfinite(prices) & (prices > 0)
    order = [PRICE_FIELDS.index(field) for field in PRICE_CONTRACT["fallback_order"]]
    first = np.asarray(order)[valid_price[:, order].argmax(axis=1)] if len(prices) else np.empty(0, np.int64)
    anchor = prices[np.arange(len(prices)), first]
    anchor = np.where(valid_price.any(axis=1), anchor, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        anchor = np.log(anchor).astype(np.float32)
    return values, anchor


def transform_frame(normalized, raw, parameters):
    base = base_transform_frame(normalized, raw)
    quantities, anchor = transform_side(raw, parameters)
    result = base[KEYS + BASE_COLUMNS[:52]].copy()
    for index, field in enumerate(QUANTITY_FIELDS):
        result["q_" + field] = quantities[:, index]
    result["log_price_reference"] = anchor
    return result[KEYS + STORAGE_COLUMNS]


def rounded(values):
    with np.errstate(over="ignore", invalid="ignore"):
        half = values.astype(np.float16).astype(np.float32)
    return np.where(np.isfinite(values) & ~np.isfinite(half), values, half)


def prepare_coordinate_windows(raw, repairs, clock, parameters):
    """raw 为每频率 (72 个 FP16 编码通道, FP32 价格中心)；解码到 112 通道。"""
    result = {}
    for frequency in FREQUENCIES:
        codes, anchors = raw[frequency]
        codes = codes.float()
        indices, values = repairs[frequency]
        if indices.numel():
            codes.reshape(-1).index_copy_(0, indices, values)
        values = codes
        length = values.shape[1]
        cutoff = PREVIOUS_DAYS[frequency] * DAILY_BARS[frequency] + clock.argmax(dim=1) * (DAILY_BARS[frequency] // 8)
        available = torch.arange(length, device=values.device)[None] < cutoff[:, None]
        finite = torch.isfinite(values) & available[..., None]
        non_book = torch.cat([finite[..., :6], finite[..., 36:42]], dim=-1).all(dim=-1)
        book = torch.cat([finite[..., 6:36], finite[..., 42:72]], dim=-1).all(dim=-1)
        flags = torch.stack([non_book, book, finite.any(dim=-1)], dim=-1).float()
        field_flags = (finite[..., :36] | finite[..., 36:72]).float()
        anchor = parameters[frequency]["price_anchor"]
        reference = (anchors.float() - float(np.float32(anchor["center"]))) / float(np.float32(anchor["scale"]))
        reference = torch.where(torch.isfinite(reference) & available, reference, 0)
        output = torch.cat([torch.where(finite, values, 0), flags, field_flags, reference[..., None]], dim=-1)
        result[frequency] = output.transpose(1, 2).contiguous()
    return result


def raw_batches(data_root, frequency, months, start, stop, *, stocks=None, batch_rows=65536):
    """小批读取授权行情，调用者负责固定来源清单和模型规则网格。"""
    columns = KEYS + list(dict.fromkeys(PRICE_FIELDS + QUANTITY_FIELDS))
    for month in months:
        path = Path(data_root) / ("cpt_jyc_2026_stock_" + frequency) / (str(month) + ".parquet")
        if not path.is_file():
            raise FileNotFoundError(path)
        predicate = (ds.field("date") >= pd.Timestamp(start).to_pydatetime()) & (ds.field("date") < pd.Timestamp(stop).to_pydatetime())
        if stocks is not None:
            predicate = predicate & ds.field("instrument").isin(list(stocks))
        scanner = ds.dataset(path, format="parquet").scanner(columns=columns, filter=predicate,
                    batch_size=batch_rows, use_threads=False)
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            frame["date"] = pd.to_datetime(frame.date)
            if frame[KEYS].isna().any().any():
                raise ValueError("原始行情键含空值")
            yield frame
