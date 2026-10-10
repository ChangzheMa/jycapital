"""平台推理入口；只读行情、股票池和冻结权重，不读取标签。"""

from pathlib import Path

import numpy as np
import pandas as pd

from v1_data import KEYS, load_month, month_intervals, platform_interval
from v1_model import NumpyMLP


def predict(datasources, start_date, end_date, model_path=None, dai_module=None):
    if dai_module is None:
        import dai as dai_module
    model_path = Path(model_path) if model_path is not None else Path(__file__).resolve().with_name("v1-model.json")
    model = NumpyMLP(model_path)
    start, stop = platform_interval(start_date, end_date)
    chunks = []
    for lo, hi in month_intervals(start, stop):
        x, keys, _ = load_month(dai_module, datasources, lo, hi, model.config)
        keys["factor"] = model.predict(x)
        chunks.append(keys[KEYS + ["factor"]])
    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=KEYS + ["factor"])
    result["date"] = pd.to_datetime(result["date"])
    result["factor"] = result["factor"].astype(np.float64)
    if result.duplicated(KEYS).any() or result[KEYS].isna().any().any() or not np.isfinite(result["factor"]).all():
        raise ValueError("推理输出违反键或数值约束")
    if not result.empty and not ((result["date"] >= start) & (result["date"] < stop)).all():
        raise ValueError("推理结果超出请求区间")
    return result.sort_values(KEYS).reset_index(drop=True)


def main(datasources, start_date, end_date):
    return predict(datasources, start_date, end_date)
