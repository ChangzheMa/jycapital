"""估算或构建三类模型共用的全量 FP16 训练缓存；不启动模型训练。"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from resident_data import TrainingPlan, build_cache, cache_key
from artifacts import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", choices=("v101", "v201", "v301"), default="v101")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "evaluation/protocols/local-v2-ic.json")
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--holdout-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    solution = ROOT / "solutions" / args.solution
    sys.path.insert(0, str(solution))
    trainer = importlib.import_module(args.solution + "_train")
    config = read_json(args.config or solution / "model-config.json")
    protocol = read_json(args.protocol)
    trainer.validate_training(config, protocol)
    if config["training"]["batching"] != "global_shuffled_samples":
        parser.error("此入口只接受全局 FP16 缓存配置")
    store = trainer.ProcessedStore(args.training_run, args.holdout_run, args.data_root, config, protocol)
    plan = TrainingPlan(store)
    estimate = plan.estimate()
    print(json.dumps({"estimate": estimate, "numeric_GiB": sum(estimate["feature_bytes"].values()) / 2**30}, ensure_ascii=False))
    if args.estimate_only:
        return
    # 大文件检查仅在实际构建阶段执行，临时元数据不会被登记成正式研究 run。
    with tempfile.TemporaryDirectory(prefix="resident-verification-", dir=args.training_run.parent) as temporary:
        scratch = Path(temporary)
        write_json(scratch / "protocol.json", protocol)
        trainer.data_manifest(store, scratch)
        key = cache_key(store, read_json(scratch / "data-manifest.json"), trainer.prepare_training_targets)
        destination = args.cache_dir or args.training_run.parent / "resident-cache" / key
        manifest = build_cache(store, destination, key, trainer.prepare_training_targets, plan=plan,
                               report=lambda event, **fields: print(json.dumps({"event": event, **fields}), flush=True))
        print(json.dumps({"cache": str(destination.resolve()), "status": manifest["status"],
                          "cache_key": key, "resident_GiB": manifest["resident_array_bytes"] / 2**30}))


if __name__ == "__main__":
    main()
