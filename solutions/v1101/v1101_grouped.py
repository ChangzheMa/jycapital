"""股票日索引与双目标视图；复用原行情数组，不复制完整样本窗口。"""

from collections import deque
import hashlib
import os
from pathlib import Path

import numpy as np

from artifacts import read_json, sha256, write_json
from v1101_data import prediction_clock_values


class StockDayIndex:
    def __init__(self, block, indices=None):
        if hasattr(block, "raw_batch"):
            total = len(block.day)
            rows = np.arange(total, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
            days, stocks, clocks = block.day[rows], block.stock[rows], block.clock[rows]
        else:
            total = len(block.keys)
            rows = np.arange(total, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
            times = block.keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            _, days = np.unique(times[rows] // 86400000000000, return_inverse=True)
            _, stocks = np.unique(block.keys.instrument.to_numpy()[rows], return_inverse=True)
            clocks = prediction_clock_values(times[rows]).argmax(axis=1)
        if not len(rows) or (rows < 0).any() or (rows >= total).any() or len(np.unique(rows)) != len(rows):
            raise ValueError("股票日视图要求非空、唯一且合法的样本索引")
        if (days < 0).any() or (stocks < 0).any() or (clocks < 0).any() or (clocks >= 8).any():
            raise ValueError("股票日索引含非法日期、股票或预测时刻")
        width = int(stocks.max()) + 1
        codes = days.astype(np.int64) * width + stocks
        unique = np.unique(codes)
        group = np.searchsorted(unique, codes)
        slots = group * 8 + clocks
        if (np.bincount(slots, minlength=len(unique) * 8) > 1).any():
            raise ValueError("同一股票日的同一预测时刻出现重复样本")
        self.samples = np.full((len(unique), 8), -1, np.int64)
        self.samples.reshape(-1)[slots] = rows
        present = self.samples >= 0
        latest = 7 - present[:, ::-1].argmax(axis=1)
        self.representatives = self.samples[np.arange(len(unique)), latest]
        self.sample_count = len(rows)

    def digest(self):
        return hashlib.sha256(self.samples.tobytes()).hexdigest()

    def assign_predictions(self, output, groups, values):
        rows = self.samples[groups]
        valid = rows >= 0
        output[rows[valid]] = values[valid]


class GroupTargets:
    def __init__(self, index, primary, ranks):
        if primary.shape != ranks.shape or not np.array_equal(np.isfinite(primary), np.isfinite(ranks)):
            raise ValueError("主目标与辅助排名目标的可用样本不一致")
        self.index, self.primary, self.ranks = index, primary, ranks
        # 双缓冲异步传输期间保留少量主机端目标数组。
        self.recent = deque(maxlen=4)

    def __getitem__(self, groups):
        rows = self.index.samples[groups]
        safe = rows.clip(0)
        values = np.stack([self.primary[safe], self.ranks[safe]], axis=-1)
        values = np.where((rows >= 0)[..., None], values, np.nan)
        self.recent.append(values)
        return values


class GroupedBlock:
    def __init__(self, base, primary, ranks):
        self.base = base
        self.index = StockDayIndex(base)
        self.targets = GroupTargets(self.index, primary, ranks)
        self.eligible = np.zeros(len(self.index.samples), bool)
        for start in range(0, len(self.eligible), 65536):
            rows = self.index.samples[start:start + 65536]
            self.eligible[start:start + len(rows)] = ((rows >= 0) & np.isfinite(primary[rows.clip(0)])).any(axis=1)

    def batch(self, indices):
        return self.base.batch(self.index.representatives[indices])


class ResidentGroupedBlock(GroupedBlock):
    def raw_batch(self, indices, destinations=None):
        return self.base.raw_batch(self.index.representatives[indices], destinations)


def training_view(base, primary, ranks):
    kind = ResidentGroupedBlock if hasattr(base, "raw_batch") else GroupedBlock
    return kind(base, primary, ranks)


def shuffled_groups(view, seed, epoch, segment):
    order = np.flatnonzero(view.eligible)
    partition = 0 if segment == "global" else int(segment.replace("-", ""))
    np.random.default_rng(np.random.SeedSequence([seed, epoch, partition])).shuffle(order)
    return order


def record_group_plan(view, run, segment):
    path = Path(run) / "stock-day-training.json"
    record = {"groups": len(view.index.samples), "eligible_groups": int(view.eligible.sum()),
              "sample_rows": view.index.sample_count,
              "eligible_sample_rows": int(np.isfinite(view.targets.primary).sum()),
              "membership_sha256": view.index.digest(),
              "primary_targets_sha256": hashlib.sha256(view.targets.primary.tobytes()).hexdigest(),
              "rank_targets_sha256": hashlib.sha256(view.targets.ranks.tobytes()).hexdigest()}
    identity = {"format": "jycapital-stock-day-training-v1",
                "data_manifest_sha256": sha256(Path(run) / "data-manifest.json"),
                "cursor_unit": "stock_day_group", "reported_sample_cursor_unit": "nominal_eight_clock_slots",
                "segments": {}}
    if path.exists():
        saved = read_json(path)
        for key in ("format", "data_manifest_sha256", "cursor_unit", "reported_sample_cursor_unit"):
            if saved.get(key) != identity[key]:
                raise ValueError("股票日训练视图的来源或游标口径改变")
        identity = saved
        if segment in saved["segments"] and saved["segments"][segment] != record:
            raise ValueError("股票日成员或监督目标在恢复时改变")
    identity["segments"][segment] = record
    temporary = path.with_suffix(".tmp")
    write_json(temporary, identity)
    os.replace(temporary, path)
    return record
