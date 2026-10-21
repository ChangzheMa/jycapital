"""复用原行情缓存，仅为当前运行创建收益与排名混合目标，不改写共享缓存。"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from artifacts import read_json, sha256, write_json
from resident_data import CLOCK_MINUTES, MINUTE_NS
from v402_data import section_indices
from v402_loss import blended_standardized_targets


def prepare_blended_training_targets(keys, protocol):
    targets = np.full(len(keys), np.nan, dtype=np.float64)
    skipped = {}
    for indices in section_indices(keys):
        labels = torch.from_numpy(keys.iloc[indices][protocol["label_column"]].to_numpy(dtype=np.float64))
        values, reason = blended_standardized_targets(labels, protocol["min_cross_section_size"])
        if reason:
            skipped[pd.Timestamp(keys.date.iloc[indices[0]]).isoformat()] = reason
        else:
            targets[indices] = values.numpy()
    return targets, skipped


def attach_blended_targets(store, run):
    """从原始标签重建目标并核对缓存行顺序；断点恢复核验独立目标指纹。"""
    original, block = store.original, store.block
    protocol = original.protocol
    interval = protocol["splits"]["train"]
    keys = original.universe.loc[
        (original.universe.date >= interval["start"]) & (original.universe.date < interval["stop"]),
        ["date", "instrument"],
    ].sort_values(["date", "instrument"]).reset_index(drop=True)
    expected_times = (block.arrays["days"][block.day].astype("datetime64[ns]").astype(np.int64)
                      + CLOCK_MINUTES[block.clock] * MINUTE_NS)
    stock_indices = pd.Index(block.manifest["stocks"]).get_indexer(keys.instrument)
    if not np.array_equal(keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64), expected_times) or \
            not np.array_equal(stock_indices, block.stock):
        raise ValueError("混合监督与基础缓存的样本键或行顺序不一致")
    labels = pq.read_table(
        original.data_root / (protocol["label_table"] + ".parquet"),
        columns=["date", "instrument", protocol["label_column"]],
        filters=[("date", ">=", pd.Timestamp(interval["start"])),
                 ("date", "<", pd.Timestamp(interval["stop"]))],
    ).to_pandas()
    keys = keys.merge(labels, on=["date", "instrument"], how="left", validate="one_to_one")
    horizon = keys.date + pd.Timedelta(minutes=protocol["label_horizon_minutes"])
    admissible = (horizon < pd.Timestamp(interval["stop"])) & (horizon.dt.normalize() == keys.date.dt.normalize())
    keys.loc[~admissible, protocol["label_column"]] = np.nan
    targets, skipped = prepare_blended_training_targets(keys, protocol)
    baseline_targets, baseline_skipped = block.training_targets()
    if not np.array_equal(np.isfinite(targets), np.isfinite(baseline_targets)) or skipped != baseline_skipped:
        raise ValueError("混合监督改变了训练样本可用性，不能复用基础缓存")
    record = {
        "format": "jycapital-blended-target-overlay-v1",
        "base_cache_key": block.manifest["cache_key"],
        "data_manifest_sha256": sha256(Path(run) / "data-manifest.json"),
        "target_transform": "cross_section_equal_return_rank_zscore",
        "tie_method": "average",
        "return_weight": 0.5,
        "rank_weight": 0.5,
        "restandardize_mixture": True,
        "label_ddof": 0,
        "rows": len(targets),
        "eligible_rows": int(np.isfinite(targets).sum()),
        "targets_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
        "target_bytes": targets.nbytes,
        "shared_input_cache_modified": False,
    }
    path = Path(run) / "target-overlay.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("混合监督的来源或目标在断点恢复时发生变化")
    write_json(path, record)
    block.targets = targets
    return record
