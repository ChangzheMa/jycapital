"""仅训练调用的下一根30分钟数量目标；不扩展因果预测输入。"""
from collections import deque
from pathlib import Path

import numpy as np

from artifacts import read_json, sha256, write_json

CONTRACT = "next_30m_z_quantity_state_v1"
INDICES = [4, 5, *range(16, 36)]
FIELDS = ["volume", "amount"] + [f"{side}_{kind}{level}" for kind in ("volume", "num_orders")
                                  for side in ("ask", "bid") for level in range(1, 6)]
GROUPS = ((0, 2), (2, 12), (12, 22))
TARGET_CHANNELS = 68


def unpack_targets(targets):
    if targets.ndim != 3 or targets.shape[1:] != (8, TARGET_CHANNELS):
        raise ValueError("训练目标必须包含原收益／排名两头及三组22维数量值")
    return targets[..., :2], targets[..., 2:24], targets[..., 24:46], targets[..., 46:68]


def _resident_quantity(base, days, stocks):
    mapping = base.arrays["bar30m-map"]
    block_ids = np.where(days >= 0, mapping[days.clip(0), stocks], 0)
    values = np.array(base.arrays["bar30m-values"][block_ids], dtype=np.float32, copy=True)
    pointer = base.arrays["bar30m-exception-pointers"]
    for row, index in enumerate(block_ids):
        start, stop = pointer[index:index + 2]
        if stop > start:
            positions = base.arrays["bar30m-exception-offsets"][start:stop]
            values[row].reshape(-1)[positions] = base.arrays["bar30m-exception-values"][start:stop]
    return values[..., INDICES]


class QuantityTargets:
    def __init__(self, base, original, protocol):
        self.base, self.original = base, original
        self.index, self.primary, self.ranks = original.index, original.primary, original.ranks
        self.recent = deque(maxlen=4)
        representatives = self.index.representatives
        if hasattr(base, "raw_batch"):
            dates = base.arrays["days"][base.day[representatives]]
        else:
            dates = base.times[representatives] // 86400000000000 * 86400000000000
        interval = protocol["splits"]["train"]
        start, stop = [np.datetime64(interval[key], "ns").astype(np.int64) for key in ("start", "stop")]
        # 所有预测时点的辅助目标都在同一交易日，最迟15:00；禁止拿留出划分构建训练视图。
        if not len(dates) or np.any(dates < start) or np.any(dates + 900 * 60000000000 >= stop):
            raise ValueError("数量辅助目标必须严格处于训练时间边界内")

    def __getitem__(self, groups):
        base = self.base
        representatives = self.index.representatives[groups]
        if hasattr(base, "raw_batch"):
            days, stocks = base.day[representatives], base.stock[representatives]
            current = _resident_quantity(base, days, stocks)
            previous = _resident_quantity(base, days - 1, stocks)
        else:
            positions = base.plans["bar30m"][1][base.day_indices[representatives]][:, -16:]
            stocks = base.stock_indices[representatives]
            grid = base.grids["bar30m"][1]
            if len(grid):
                raw = grid[positions.clip(0), stocks[:, None]][..., INDICES].copy()
                raw[positions < 0] = np.nan
            else:
                raw = np.full((len(groups), 16, 22), np.nan, np.float32)
            if base.config["data"]["input_precision"] == "fp16_with_fp32_overflow":
                with np.errstate(over="ignore", invalid="ignore"):
                    half = raw.astype(np.float16).astype(np.float32)
                raw = np.where(np.isfinite(raw) & ~np.isfinite(half), raw, half)
            previous, current = raw[:, :8], raw[:, 8:]
        latest = np.concatenate([previous[:, -1:], current[:, :-1]], axis=1)
        present = self.index.samples[groups] >= 0
        pieces = [self.original[groups]] + [np.where(present[..., None], values, np.nan)
                                          for values in (current, latest, previous)]
        result = np.concatenate(pieces, axis=-1).astype(np.float64, copy=False)
        self.recent.append(result)
        return result


def attach_quantity_targets(view, protocol):
    view.targets = QuantityTargets(view.base, view.targets, protocol)
    return view


def record_quantity_plan(view, run, segment):
    path = Path(run) / "quantity-supervision.json"
    base = view.base
    record = {"contract": CONTRACT, "source_sha256": sha256(__file__),
              "data_manifest_sha256": sha256(Path(run) / "data-manifest.json"),
              "fields": FIELDS, "source_z_indices": INDICES, "field_groups": [list(v) for v in GROUPS],
              "target_channels": TARGET_CHANNELS, "train_only": True,
              "future_inference_input": False, "provided_return_targets_modified": False,
              "target_bar_ends": ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"],
              "known_baselines": ["last_closed_bar", "previous_day_same_slot"], "segments": {}}
    segment_record = {"membership_sha256": view.index.digest(), "groups": len(view.index.samples),
                      "eligible_groups": int(view.eligible.sum()), "extra_full_array_bytes": 0}
    if hasattr(base, "raw_batch"):
        segment_record.update(cache_key=base.manifest["cache_key"],
                              source_files={name: value for name, value in base.manifest["files"].items()
                                            if name.startswith("bar30m-") or name in ("days.npy", "sample_day.npy", "sample_stock.npy", "sample_clock.npy")})
    else:
        segment_record["source"] = "frozen_native_training_grid"
    if path.exists():
        previous = read_json(path)
        for key, value in record.items():
            if key != "segments" and previous.get(key) != value:
                raise ValueError("数量监督契约或来源在恢复时改变")
        record = previous
        if segment in record["segments"] and record["segments"][segment] != segment_record:
            raise ValueError("数量监督成员或输入来源在恢复时改变")
    record["segments"][segment] = segment_record
    write_json(path.with_suffix(".tmp"), record)
    path.with_suffix(".tmp").replace(path)
    return record
