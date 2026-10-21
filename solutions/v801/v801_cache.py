"""冻结本地父模型的 FP32 表示缓存；仅用于本地训练，不进入平台材料。"""

import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time

import torch
import numpy as np
import pandas as pd

from artifacts import read_json, write_json, sha256, log_event
from resident_data import open_resident_store, ResidentStore, ResidentBatches, CLOCK_MINUTES, MINUTE_NS
from evaluation_cache import cache_key as evaluation_key, build_evaluation_cache, EvaluationData
from v801_model import MultiFrequencyGRU
from v801_text_artifact import state_digest, decode_state

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "jycapital-frozen-representation-v1"


def initialize_backbone(model, config, artifact=None):
    spec = config["initialization"]
    if artifact is None:
        path = ROOT / "outputs/runs" / spec["run_id"] / "best.pt"
        if sha256(path) != spec["model_sha256"]:
            raise ValueError("冻结父模型权重指纹不符")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("weight_source") != "ema" or checkpoint["config"]["data"] != config["data"]:
            raise ValueError("v801 要求本地父版本 EMA 与同一输入配置")
        state = checkpoint["state_dict"]
    else:
        state = decode_state(read_json(artifact))
        checkpoint = None
    if state_digest(state) != spec["state_dict_sha256"]:
        raise ValueError("冻结父模型的张量内容不符")
    model.backbone.load_state_dict(state, strict=True)
    model.backbone.requires_grad_(False).eval()
    return checkpoint


def group_rows(block, rows):
    days, stocks, clocks = block.day[rows], block.stock[rows], block.clock[rows]
    codes = days.astype(np.int64) * (int(block.stock.max()) + 1) + stocks
    _, inverse = np.unique(codes, return_inverse=True)
    slots = inverse * 8 + clocks
    if (np.bincount(slots) > 1).any():
        raise ValueError("表示缓存含重复股票日时点")
    members = np.full((int(inverse.max()) + 1, 8), -1, np.int64)
    members.reshape(-1)[slots] = rows
    latest = 7 - (members[:, ::-1] >= 0).argmax(1)
    representatives = members[np.arange(len(members)), latest]
    return members, representatives


class RepresentationBlock:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        if self.manifest["identity"] != identity or self.manifest["status"] != "complete":
            raise ValueError("表示缓存来源或状态不符")
        for name, record in self.manifest["files"].items():
            path = self.directory / name
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise ValueError("表示缓存文件改变: " + name)
        self.keys = pd.read_parquet(self.directory / "keys.parquet")
        self.features = np.load(self.directory / "features.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(self.directory / "targets.npy", mmap_mode="r", allow_pickle=False)
        if self.features.shape != (len(self.keys), self.manifest["feature_dim"]) or self.features.dtype != np.float32:
            raise ValueError("表示缓存维度或精度不符")
        if self.targets.shape != (len(self.keys),) or self.targets.dtype != np.float64:
            raise ValueError("表示缓存目标维度或精度不符")
        times = self.keys.date.to_numpy(dtype="datetime64[ns]")
        if self.keys.duplicated(["date", "instrument"]).any() or (times[1:] < times[:-1]).any():
            raise ValueError("表示缓存键重复或时间顺序改变")
        self.boundaries = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1, len(times)]

    def training_targets(self):
        catalog = self.manifest["catalog"]
        return self.targets, {stamp: item["reason"] for stamp, item in catalog["skipped"].items()}

    def section(self, number):
        return np.arange(self.boundaries[number], self.boundaries[number + 1])

    def close(self):
        for array in (self.features, self.targets):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()


class RepresentationStore:
    representation_cache = True

    def __init__(self, original, run, transform, resident_directory=None, backbone_artifact=None):
        self.original, self.run, self.transform = original, Path(run), transform
        self.config, self.protocol = original.config, original.protocol
        self.resident_directory = resident_directory
        self.backbone_artifact = backbone_artifact
        self.device = torch.device("cpu")
        self.loaded = {}

    def months(self, split):
        return ["global"]

    def close(self):
        for block in self.loaded.values():
            block.close()
        self.loaded.clear()

    def load(self, split, segment):
        if segment != "global":
            raise ValueError("冻结表示使用全局分区")
        if split in self.loaded:
            return self.loaded[split]
        module = Path(__file__).resolve().parent
        identity = {"format": FORMAT, "split": split, "protocol": self.protocol,
                    "data_manifest_sha256": sha256(self.run / "data-manifest.json"),
                    "data": self.config["data"], "parent": self.config["initialization"],
                    "encoder_function": inspect.getsource(MultiFrequencyGRU.encode_shared),
                    "encoding_environment": {"torch": str(torch.__version__), "device": str(self.device)},
                    "packing_sources": {name: sha256(ROOT / "common" / name)
                                        for name in ("resident_data.py", "evaluation_cache.py")},
                    "encoding_sources": {name: sha256(module / name) for name in (
                        "v801_backbone.py", "v801_encoding.py", "v801_schema.py", "v801_data.py", "v801_cache.py")}}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        directory = self.original.training_run.parent / "representation-cache" / key
        started = time.perf_counter()
        cache_hit = directory.exists()
        if not cache_hit:
            self.build(split, directory, identity)
        block = RepresentationBlock(directory, identity)
        record_path = self.run / "representation-cache.json"
        record = read_json(record_path) if record_path.exists() else {"format": FORMAT, "splits": {}}
        current = {"cache_key": key, "directory": str(directory.resolve()),
                   "manifest_sha256": sha256(directory / "manifest.json"),
                   "features_sha256": block.manifest["files"]["features.npy"]["sha256"],
                   "rows": len(block.keys), "feature_dim": block.features.shape[1]}
        if split in record["splits"] and record["splits"][split] != current:
            raise ValueError("当前运行的冻结表示缓存发生变化")
        record["splits"][split] = current
        execution = {"cache_hit": cache_hit, "open_and_build_seconds": time.perf_counter() - started,
                     "feature_matrix_bytes": int(block.features.nbytes)}
        record.setdefault("execution", {})[split] = execution
        write_json(record_path, record)
        self.loaded[split] = block
        log_event(self.run, "representation_cache_ready", split=split, rows=len(block.keys),
                  feature_dim=block.features.shape[1], precision="float32", cache_key=key, **execution)
        return block

    def build(self, split, directory, identity):
        # 构造缓存模型不能消耗正式训练的随机状态，缓存命中与重建须得到同一训练序列。
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            model = MultiFrequencyGRU(self.config).to(self.device).eval()
            initialize_backbone(model, self.config, self.backbone_artifact)
        original = self.original
        evaluation = None
        catalog = None
        if split == "train":
            resident = open_resident_store(original, self.run, self.transform, self.resident_directory,
                                          lambda event, **fields: log_event(self.run, event, **fields))
            base = resident.block
            interval = self.protocol["splits"][split]
            keys = original.universe.loc[(original.universe.date >= interval["start"]) &
                                          (original.universe.date < interval["stop"]), ["date", "instrument"]]
            keys = keys.sort_values(["date", "instrument"]).reset_index(drop=True)
            expected = base.arrays["days"][base.day].astype("datetime64[ns]").astype(np.int64) + CLOCK_MINUTES[base.clock] * MINUTE_NS
            if not np.array_equal(keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64), expected) or not np.array_equal(
                    pd.Index(base.manifest["stocks"]).get_indexer(keys.instrument), base.stock):
                raise ValueError("冻结表示的训练键与原缓存行顺序不同")
            catalog = base.manifest["catalog"]
            segments = [("global", np.arange(len(keys)))]
        else:
            key = evaluation_key(original, split, read_json(self.run / "data-manifest.json"))
            cache = original.training_run.parent / "evaluation-cache" / key
            build_evaluation_cache(original, split, cache, key,
                                   lambda event, **fields: log_event(self.run, "evaluation_" + event, split=split, **fields))
            evaluation = EvaluationData(cache, key, getattr(self, "evaluation_cache_mode", "stream"),
                                        self.config["training"]["resident_cache"]["memory_reserve_gib"],
                                        self.config["inference"]["stock_day_batch_size"])
            keys = evaluation.keys
            layout = ResidentStore(original, None, self.config["training"]["resident_cache"]["evaluation_days"])
            times = keys.date.to_numpy(dtype="datetime64[ns]")
            segments = [(name, np.arange(np.searchsorted(times, np.datetime64(lo)), np.searchsorted(times, np.datetime64(hi))))
                        for name, (lo, hi) in layout.segments[split].items()]
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=directory.name[:12] + "-", dir=directory.parent) as temporary:
                staging = Path(temporary)
                features = np.lib.format.open_memmap(staging / "features.npy", mode="w+", dtype=np.float32,
                                                     shape=(len(keys), model.feature_dim))
                count = 0
                with torch.inference_mode():
                    for name, rows in segments:
                        block = base if split == "train" else evaluation.load_segment(rows)
                        members, representatives = group_rows(block, rows)
                        iterator = ResidentBatches(block, representatives, block.targets, 0,
                                                   self.config["inference"]["stock_day_batch_size"], self.device)
                        for offset, selected, windows, clocks, _ in iterator:
                            encoded = model.encode_shared({f: x.contiguous() for f, x in windows.items()}, clocks).cpu().numpy()
                            targets = members[offset:offset + len(selected)]
                            present = targets >= 0
                            features[targets[present]] = encoded[present]
                            count += int(present.sum())
                        del iterator, block, members, representatives
                        log_event(self.run, "representation_encoded", split=split, segment=name, rows=len(rows))
                if count != len(keys):
                    raise ValueError("表示缓存没有覆盖完整股票池")
                features.flush()
                for offset in range(0, len(keys), 65536):
                    if not np.isfinite(features[offset:offset + 65536]).all():
                        raise FloatingPointError("非有限冻结表示")
                del features
                keys.to_parquet(staging / "keys.parquet", index=False)
                np.save(staging / "targets.npy", base.targets if split == "train" else np.zeros(len(keys), np.float64))
                files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in staging.iterdir() if p.is_file()}
                write_json(staging / "manifest.json", {"status": "complete", "identity": identity,
                           "feature_dim": model.feature_dim, "catalog": catalog, "files": files})
                os.rename(staging, directory)
        finally:
            if evaluation is not None:
                evaluation.close()
