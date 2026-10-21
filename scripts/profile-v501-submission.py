"""在 CPU 上测量独立 v501 提交包的阶段耗时；不训练、不读取评估标签。"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def profile(package, start, end, output, predictions=None):
    import dai
    package = Path(package).resolve()
    notebook = json.loads((package / "v501-submission.ipynb").read_text(encoding="utf-8"))
    namespace = {"__name__": "submission_profile", "__file__": str(package / "v501-submission.ipynb")}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "v501-submission.ipynb", "exec"), namespace)
    started = time.perf_counter()
    bundle, modules = namespace["_runtime"]()
    load_seconds = time.perf_counter() - started
    predictor = modules["v501_predict"]
    timings = {}

    def instrument(owner, name, label):
        original = getattr(owner, name)
        def measured(*args, **kwargs):
            begin = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                row = timings.setdefault(label, {"calls": 0, "seconds": 0.0})
                row["calls"] += 1
                row["seconds"] += time.perf_counter() - begin
        setattr(owner, name, measured)

    instrument(predictor, "read_table", "read_table_including_query_and_frame_checks")
    instrument(predictor, "warm_history", "warm_history_including_its_queries")
    instrument(predictor, "prior_trading_days", "prior_trading_days")
    instrument(predictor.Normalizer, "transform", "normalize")
    instrument(predictor.WindowBlock, "__init__", "window_plan")
    instrument(predictor.WindowBlock, "batch", "window_batch")
    instrument(predictor.MultiFrequencyGRU, "forward", "original_model_forward")
    if "v501_fast" in modules:
        instrument(predictor, "predict_day", "shared_day_including_windows_and_model")
        instrument(modules["v501_fast"], "predict_all_clocks", "shared_model_forward")
    started = time.perf_counter()
    result = predictor.predict(
        {f: "cpt_jyc_2026_stock_" + f for f in ("bar1m", "bar5m", "bar30m")},
        start, end, bundle=bundle, dai_module=dai, device="cpu")
    seconds = time.perf_counter() - started
    if result.empty or not np.isfinite(result.factor).all() or result.duplicated(["date", "instrument"]).any():
        raise ValueError("推理结果为空、非有限或键重复")
    if predictions:
        predictions = Path(predictions).resolve()
        if not predictions.is_relative_to(ROOT / "outputs"):
            raise ValueError("预测仅可保存到忽略的 outputs 目录")
        predictions.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(predictions, index=False)
    memory = psutil.Process().memory_info()
    report = {"checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "package": package.relative_to(ROOT).as_posix(), "run_id": bundle["metadata"]["run_id"],
              "package_sha256": hashlib.sha256((package / "v501-model.json").read_bytes()).hexdigest(),
              "model_sha256": bundle["metadata"]["selection"]["model_sha256"],
              "device": "cpu", "cpu_threads": torch.get_num_threads(), "torch": torch.__version__,
              "start": start, "end": end, "rows": len(result), "bundle_load_seconds": load_seconds,
              "prediction_seconds": seconds, "inclusive_stage_timings": timings,
              "process_peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
              "inference_execution": bundle.get("inference_execution", {"method": "original_per_sample"}),
              "scope": "本地 dai 兼容层、CPU、完整请求；阶段耗时存在嵌套，不能全部相加；不代表平台耗时。"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--start", default="2024-12-30")
    parser.add_argument("--end", default="2024-12-30")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    profile(args.package, args.start, args.end, args.output, args.predictions)
