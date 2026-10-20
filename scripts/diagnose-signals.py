"""对已完成冻结预测计算日内换手及持续性代理，保留固定本地评估指标。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "common", ROOT / "evaluation"):
    sys.path.insert(0, str(directory))
from artifacts import now, read_json, sha256, write_json
from signal_diagnostics import intraday_diagnostics
from compare import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, correlations = [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        compare(args.runs, split, allow_selection_metric_change=True)
        previous = {}
        for run in args.runs:
            manifest = read_json(run / "manifest.json")
            path = run / "evaluation" / split / "predictions.parquet"
            predictions = pd.read_parquet(path).sort_values(["date", "instrument"]).reset_index(drop=True)
            summary, transitions = intraday_diagnostics(predictions)
            directory = args.output.parent / run.name / split
            directory.mkdir(parents=True, exist_ok=True)
            transitions.to_csv(directory / "signal-transitions.csv", index=False, encoding="utf-8-sig")
            values = ["rank_autocorrelation", "one_way_turnover", "long_retention", "short_retention"]
            monthly = transitions.groupby(transitions.date.dt.strftime("%Y-%m"))[values].mean()
            clock = transitions.groupby(transitions.date.dt.strftime("%H:%M"))[values].mean()
            rows.append({"run_id": run.name, "solution_id": manifest["solution_id"], "split": split,
                         "predictions_sha256": sha256(path), "summary": summary,
                         "monthly": monthly.reset_index(names="month").to_dict("records"),
                         "clock": clock.reset_index(names="clock").to_dict("records"),
                         "local_metrics": read_json(run / "evaluation" / split / "metrics.json")})
            for name, before in previous.items():
                merged = before.merge(predictions, on=["date", "instrument"], how="outer", suffixes=("_old", "_new"), validate="one_to_one")
                if merged[["factor_old", "factor_new"]].isna().any().any():
                    raise ValueError("预测键不一致，不能直接计算配对相关性")
                grouped = merged.groupby("date")
                centered = merged[["factor_old", "factor_new"]] - grouped[["factor_old", "factor_new"]].transform("mean")
                numerator = (centered.factor_old * centered.factor_new).groupby(merged.date).sum()
                denominator = np.sqrt(centered.pow(2).groupby(merged.date).sum().prod(axis=1))
                correlation = numerator / denominator.where(denominator > 0)
                correlations.append({"left": name, "right": run.name, "split": split,
                                     "mean_cross_section_prediction_pearson": float(correlation.mean()),
                                     "undefined_sections": int(correlation.isna().sum())})
            previous[run.name] = predictions
    result = {"schema_version": 1, "computed_at": now(), "method": {
        "name": "intraday_decile_turnover_proxy_v1", "tail_fraction": 0.1,
        "weights": "top +0.5 / k, bottom -0.5 / k; ties at selection boundary share remaining mass equally",
        "gross_exposure": 1, "turnover": "0.5 * sum(abs(weight_t - weight_previous)); both portfolios must have gross 1",
        "transitions": "adjacent observed timestamps within the same local trading date; no overnight transition",
        "stock_changes": "weights align on union with zero for absent names; rank persistence uses the intersection",
        "rank_persistence": "Pearson correlation of average ranks across consecutive cross-sections",
        "label_usage": "none in signal/position diagnostics; existing local metrics retain original evaluator",
        "source_sha256": sha256(ROOT / "evaluation/signal_diagnostics.py")},
        "results": rows, "prediction_correlations": correlations,
        "limitations": ["仅本地未中性化预测的诊断代理，不是平台换手、BARRA、夏普、成本或综合分。",
                        "不评估组合收益，不据测试期调参或构造按时点手工切换规则。",
                        "单种子；验证和测试均属于持续观察的研究区间。"]}
    write_json(args.output, result)
    for row in rows:
        print(json.dumps({"solution_id": row["solution_id"], "split": row["split"], **row["summary"]}))


if __name__ == "__main__":
    main()
