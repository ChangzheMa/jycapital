"""复用父几何和FP32价格中心，进行流式或常驻评估。"""
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import psutil

from artifacts import read_json, sha256
import geometry_evaluation_cache as base_evaluation
from trade_reference_inputs import FREQUENCIES, base_data_manifest
from trade_reference_resident import ResidentBlock, ensure_side_cache, require_memory


class EvaluationData:
    def __init__(self, store, split, manifest, mode="auto", report=None):
        if mode not in ("auto", "resident", "stream"):
            raise ValueError("未知评估缓存模式")
        key = base_evaluation.cache_key(store.base, split, base_data_manifest(manifest))
        base_directory = store.training_run.parent / "evaluation-cache" / key
        base_evaluation.build_evaluation_cache(store.base, split, base_directory, key, report)
        parent = read_json(base_directory / "evaluation.json")
        if sha256(base_directory / "data/manifest.json") != parent["packed_manifest_sha256"]:
            raise ValueError("基础评测缓存清单改变")
        self.directory, self.key = ensure_side_cache(store, base_directory / "data", key, manifest, split=split, report=report)
        saved = read_json(self.directory / "manifest.json")
        spec = store.config["training"]["resident_cache"]
        self.reserve_gib = spec["memory_reserve_gib"]
        self.buffer_bytes = store.config["inference"]["stock_day_batch_size"] * sum(store.config["data"]["window_bars"].values()) * 83 * 4 * 3
        require_memory(self.buffer_bytes, self.reserve_gib)
        enough = psutil.virtual_memory().available >= saved["resident_array_bytes"] + self.buffer_bytes + self.reserve_gib * 2**30
        resident = mode == "resident" or (mode == "auto" and enough)
        if resident: require_memory(saved["resident_array_bytes"] + self.buffer_bytes, self.reserve_gib)
        self.mode = "resident" if resident else "stream"
        if sha256(base_directory / "keys.parquet") != parent["keys_sha256"]:
            raise ValueError("基础评测键或标签改变")
        self.keys = pd.read_parquet(base_directory / "keys.parquet")
        self.block = ResidentBlock(self.directory, self.key, self.reserve_gib, report, resident=resident)
        self.block.keys = self.keys
        if len(self.keys) != len(self.block.targets):
            self.close()
            raise ValueError("评测键与打包样本数不同")

    def load_segment(self, order):
        order = np.asarray(order)
        if order.ndim != 1 or (order < 0).any() or (order >= len(self.keys)).any():
            raise ValueError("非法评估分块索引")
        require_memory(self.buffer_bytes, self.reserve_gib)
        return self.block

    def close(self):
        if getattr(self, "block", None) is not None:
            self.block.close()
            self.block = None


def open_evaluation_data(store, split, manifest, mode="auto", report=None):
    return EvaluationData(store, split, manifest, mode, report)
