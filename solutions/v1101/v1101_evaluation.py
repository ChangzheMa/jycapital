"""复用冻结行情缓存的股票日评估，不修改公共缓存格式或缓存键。"""

from pathlib import Path
import time

import torch
import numpy as np

from artifacts import read_json, log_event
from baseline_evaluation import evaluate
from v1101_evaluation_cache import cache_key, build_evaluation_cache, EvaluationData
import v1101_resident as resident
from v1101_data import KEYS
from v1101_grouped import StockDayIndex


def predict_cached_split(store, split, model, device, config, run):
    """完整评估键按股票日共享编码；所有因子仅取收益主头。"""
    started = time.perf_counter()
    mode = getattr(store, "evaluation_cache_mode", "auto")
    active = getattr(store, "evaluation_cache", None)
    if active is None or active[0] != split:
        if active is not None:
            active[1].close()
            store.evaluation_cache = None
            del active
        original = store.original
        key = cache_key(original, split, read_json(Path(run) / "data-manifest.json"))
        directory = original.training_run.parent / "evaluation-cache" / key
        report = lambda event, **fields: log_event(run, "evaluation_" + event, split=split, **fields)
        build_evaluation_cache(original, split, directory, key, report)
        data = EvaluationData(directory, key, mode, config["training"]["resident_cache"]["memory_reserve_gib"],
                              config["inference"]["stock_day_batch_size"])
        store.evaluation_cache = (split, data)
        log_event(run, "evaluation_cache_ready", split=split, mode=data.mode, cache=str(directory.resolve()),
                  rows=len(data.keys), seconds=time.perf_counter() - started)
    data = store.evaluation_cache[1]
    # 只改计算组织，完整股票池、标签与分块日期边界保持一致。
    times = data.keys.date.to_numpy(dtype="datetime64[ns]")
    segments = [(name, np.arange(np.searchsorted(times, np.datetime64(lo)),
                                np.searchsorted(times, np.datetime64(hi))))
                for name, (lo, hi) in store.segments[split].items()]
    predictions = np.empty(len(data.keys), np.float64)
    model.eval()
    prediction_started = time.perf_counter()
    with torch.inference_mode():
        for number, (name, order) in enumerate(segments):
            block = data.load_segment(order)
            grouping = StockDayIndex(block, order)
            iterator = resident.ResidentBatches(block, grouping.representatives, block.targets, 0,
                                               config["inference"]["stock_day_batch_size"], device)
            for offset, indices, windows, clock, _ in iterator:
                # 每个代表行对应一个股票日；回填该组实际请求的时点。
                windows = {f: x.contiguous() for f, x in windows.items()}
                scores = (model.forward_group(windows, clock)[..., 0] /
                          config["inference"]["factor_scale"]).cpu().numpy()
                groups = np.arange(offset, offset + len(indices))
                grouping.assign_predictions(predictions, groups, scores)
            del iterator, block
            log_event(run, "predicted_month", split=split, month=name, rows=len(order))
    frame = data.keys[KEYS].copy()
    frame["factor"] = predictions
    metrics_started = time.perf_counter()
    result = evaluate(frame, data.keys, store.protocol)
    log_event(run, "cached_evaluation_finished", split=split, mode=data.mode,
              prediction_seconds=metrics_started - prediction_started,
              metric_seconds=time.perf_counter() - metrics_started, seconds=time.perf_counter() - started)
    return frame, data.keys, result
