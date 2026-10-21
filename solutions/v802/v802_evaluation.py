"""每轮重算可训练时序表示，收齐完整时点后执行集合交互。"""

from pathlib import Path
import time

import torch
import numpy as np

from artifacts import read_json, log_event
from baseline_evaluation import evaluate
from evaluation_cache import cache_key, build_evaluation_cache, EvaluationData
from resident_data import ResidentBatches
from v802_data import KEYS


def predict_cached_split(store, split, model, device, config, run):
    started = time.perf_counter()
    active = getattr(store, "evaluation_cache", None)
    if active is None or active[0] != split:
        if active is not None:
            active[1].close()
            store.evaluation_cache = None
            del active
        original = store.original
        key = cache_key(original, split, read_json(Path(run) / "data-manifest.json"))
        directory = original.training_run.parent / "evaluation-cache" / key
        build_evaluation_cache(original, split, directory, key,
                               lambda event, **fields: log_event(run, "evaluation_" + event, split=split, **fields))
        data = EvaluationData(directory, key, getattr(store, "evaluation_cache_mode", "stream"),
                              config["training"]["resident_cache"]["memory_reserve_gib"],
                              config["inference"]["stock_day_batch_size"])
        store.evaluation_cache = (split, data)
        log_event(run, "evaluation_cache_ready", split=split, mode=data.mode, cache=str(directory.resolve()),
                  rows=len(data.keys), seconds=time.perf_counter() - started)
    data = store.evaluation_cache[1]
    times = data.keys.date.to_numpy(dtype="datetime64[ns]")
    scores = np.empty(len(times), np.float64)
    model.eval()
    prediction_started = time.perf_counter()
    with torch.inference_mode():
        for name, (lo, hi) in store.segments[split].items():
            low, high = np.searchsorted(times, np.datetime64(lo)), np.searchsorted(times, np.datetime64(hi))
            rows = np.arange(low, high)
            block = data.load_segment(rows)
            codes = block.day[rows].astype(np.int64) * (int(block.stock.max()) + 1) + block.stock[rows]
            _, group = np.unique(codes, return_inverse=True)
            members = np.full((int(group.max()) + 1, 8), -1, np.int64)
            members[group, block.clock[rows]] = rows
            latest = 7 - (members[:, ::-1] >= 0).argmax(axis=1)
            representatives = members[np.arange(len(members)), latest]
            features = np.empty((len(rows), model.feature_dim), np.float32)
            iterator = ResidentBatches(block, representatives, block.targets, 0,
                                       config["inference"]["stock_day_batch_size"], device)
            for offset, selected, windows, clock, _ in iterator:
                encoded = model.encode_shared({f: x.contiguous() for f, x in windows.items()}, clock).cpu().numpy()
                targets = members[offset:offset + len(selected)]
                present = targets >= 0
                features[targets[present] - low] = encoded[present]
            section_ends = np.r_[0, np.flatnonzero(times[low + 1:high] != times[low:high - 1]) + 1, len(rows)]
            for start, stop in zip(section_ends[:-1], section_ends[1:]):
                values = torch.from_numpy(features[start:stop]).to(device)
                scores[low + start:low + stop] = (
                    model.forward_features(values) / config["inference"]["factor_scale"]).cpu().numpy()
            del iterator, block, features, members
            log_event(run, "predicted_month", split=split, month=name, rows=len(rows))
    frame = data.keys[KEYS].copy()
    frame["factor"] = scores
    metrics_started = time.perf_counter()
    result = evaluate(frame, data.keys, store.protocol)
    log_event(run, "cached_evaluation_finished", split=split, mode=data.mode,
              prediction_seconds=metrics_started - prediction_started,
              metric_seconds=time.perf_counter() - metrics_started, seconds=time.perf_counter() - started)
    return frame, data.keys, result
