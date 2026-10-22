"""共享价格坐标输入契约；只使用已闭合 Bar 的原始价格，不构造收益标签。"""
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["date", "instrument"]
FIELDS = ["open", "high", "low", "close", "volume", "amount"] + [
    f"{prefix}{level}" for prefix in ("ask_price", "bid_price", "ask_volume", "bid_volume",
                                     "ask_num_orders", "bid_num_orders") for level in range(1, 6)
]
PRICE_FIELDS = [field for field in FIELDS if field in ("open", "high", "low", "close") or "price" in field]
BASE_COLUMNS = [kind + "_" + field for kind in ("z", "c") for field in FIELDS]
VALUE_COLUMNS = ["z_" + field for field in FIELDS] + [
    ("g_" if field in PRICE_FIELDS else "c_") + field for field in FIELDS
]
CONTRACT = {
    "version": "closed_bar_shared_price_center_v1",
    "price_view": "100_log_price_minus_log_anchor",
    "anchor": "same_closed_bar_close_else_first_valid_price",
    "fallback_order": ["close"] + [field for field in PRICE_FIELDS if field != "close"],
    "long_view": "unchanged_frozen_z",
    "quantity_view": "unchanged_frozen_cross_section_c",
    "invalid_price": "nonpositive_or_nonfinite_is_missing",
}
NORMALIZATION = "frozen_z_quantity_c_and_shared_price_geometry"


def validate_geometry_config(config):
    if config["data"].get("normalization") != NORMALIZATION or config["data"].get("price_geometry") != CONTRACT:
        raise ValueError("共同价格坐标输入契约不匹配")


def price_view(raw):
    """每行共用一个价格中心；行内和跨行均不依赖后续 Bar。"""
    prices = raw[PRICE_FIELDS].to_numpy(dtype=np.float64)
    valid = np.isfinite(prices) & (prices > 0)
    order = [PRICE_FIELDS.index(name) for name in CONTRACT["fallback_order"]]
    first = np.asarray(order)[valid[:, order].argmax(axis=1)] if len(prices) else np.empty(0, np.int64)
    anchors = prices[np.arange(len(prices)), first]
    anchors = np.where(valid.any(axis=1), anchors, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 100 * (np.log(prices) - np.log(anchors[:, None]))
    return np.where(valid, values, np.nan).astype(np.float32)


def transform_frame(normalized, raw):
    """保留 z 和数量 c，仅替换第二视图中的价格字段。"""
    normalized, raw = normalized.reset_index(drop=True), raw.reset_index(drop=True)
    aligned = len(normalized) == len(raw)
    if aligned and len(raw):
        aligned = (np.array_equal(pd.to_datetime(normalized.date).to_numpy(dtype="datetime64[ns]"),
                                  pd.to_datetime(raw.date).to_numpy(dtype="datetime64[ns]")) and
                   np.array_equal(normalized.instrument.astype(str).to_numpy(), raw.instrument.astype(str).to_numpy()))
    if not aligned:
        raise ValueError("原始价格与预处理键必须按同一顺序一对一对应")
    if normalized[KEYS].isna().any().any() or raw[KEYS].isna().any().any() or normalized.duplicated(KEYS).any():
        raise ValueError("价格表达转换包含空键或重复键")
    result = normalized[KEYS + BASE_COLUMNS].rename(columns={"c_" + f: "g_" + f for f in PRICE_FIELDS}).copy()
    values = price_view(raw)
    for column, field in enumerate(PRICE_FIELDS):
        result["g_" + field] = values[:, column]
    return result[KEYS + VALUE_COLUMNS]


class RawPriceSource:
    """至多缓存一个原始月份的价格索引；不落地新的归一化 Parquet。"""

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.identity, self.indexed = None, None

    def close(self):
        self.identity, self.indexed = None, None

    def transform(self, frame, frequency):
        if frequency not in ("bar1m", "bar5m", "bar30m"):
            raise ValueError("原始价格读取只接受登记的三种行情频率")
        if frame.empty:
            raw = frame[KEYS].copy()
            for field in PRICE_FIELDS:
                raw[field] = pd.Series(dtype=np.float32)
            return transform_frame(frame, raw)
        months = frame.date.dt.to_period("M").unique()
        if len(months) != 1:
            raise ValueError("原始价格读取器每批只接受一个月份")
        identity = frequency, str(months[0])
        if identity != self.identity:
            self.close()
            import pyarrow.parquet as pq
            import psutil
            source = self.data_root / ("cpt_jyc_2026_stock_" + frequency) / (identity[1] + ".parquet")
            rows = pq.ParquetFile(source).metadata.num_rows
            estimated = rows * 192 + 1024 ** 3
            if psutil.virtual_memory().available < estimated:
                raise MemoryError("单月原始价格索引缺少内存余量")
            raw = pd.read_parquet(source, columns=KEYS + PRICE_FIELDS)
            raw["date"] = pd.to_datetime(raw.date)
            if raw[KEYS].isna().any().any() or raw.duplicated(KEYS).any():
                raise ValueError("原始价格月份包含空键或重复键")
            self.indexed = raw.set_index(KEYS)
            self.identity = identity
        keys = pd.MultiIndex.from_frame(frame[KEYS])
        positions = self.indexed.index.get_indexer(keys)
        if (positions < 0).any():
            raise ValueError("原始价格缺少预处理键；禁止静默删行或填造价格")
        raw = self.indexed.iloc[positions].reset_index()
        return transform_frame(frame, raw)


def verified_raw_sources(store):
    """原始价格须与已有预处理对应同一行情快照，纳入新运行的数据指纹。"""
    from artifacts import read_json, sha256
    records = {}
    for split in ("train", "validation", "test"):
        prepared = store.training_run if split == "train" else store.holdout_run
        original = read_json(prepared / "data-manifest.json")["files"]
        for frequency in ("bar1m", "bar5m", "bar30m"):
            for month in store.months(split):
                relative = f"cpt_jyc_2026_stock_{frequency}/{month}.parquet"
                if relative in records:
                    continue
                expected = original[relative]
                path = store.data_root / relative
                digest = sha256(path)
                if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
                    raise ValueError("原始价格与冻结预处理来源不一致: " + relative)
                records[relative] = {"path": "raw/" + relative, "bytes": path.stat().st_size, "sha256": digest}
    return [records[key] for key in sorted(records)]
