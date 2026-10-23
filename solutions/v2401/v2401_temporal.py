"""由固定训练边界生成股票日损失权重，保留全部历史样本。"""

import hashlib
import numpy as np


def attach_temporal_weights(view, config, protocol):
    spec = config["training"]["temporal_weighting"]
    rows = view.index.representatives
    if hasattr(view.base, "arrays"):
        dates = np.asarray(view.base.arrays["days"])[view.base.day[rows]].astype("datetime64[ns]").astype("datetime64[D]")
    else:
        dates = view.base.keys.iloc[rows].date.to_numpy(dtype="datetime64[ns]").astype("datetime64[D]")
    interval = protocol["splits"]["train"]
    start, stop = np.datetime64(interval["start"], "D"), np.datetime64(interval["stop"], "D")
    if np.isnat(dates).any() or (dates < start).any() or (dates >= stop).any():
        raise ValueError("时间加权只允许固定训练区间内的日期")
    age = (stop - dates).astype(np.float64)
    weights = spec["floor"] + (1.0 - spec["floor"]) * np.exp2(-age / spec["half_life_days"])
    view.sample_weights = weights.astype(np.float64)
    view.temporal_weighting_record = {
        "policy": dict(spec), "training_interval": dict(interval),
        "dates_sha256": hashlib.sha256(dates.tobytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(view.sample_weights.tobytes()).hexdigest(),
        "groups": len(weights), "minimum": float(weights.min()), "maximum": float(weights.max()),
        "mean": float(weights.mean()), "normalization": "sum_of_valid_sample_weights_within_each_batch",
    }
    return view
