"""将已有完整截面 IC 分解为真实标签幅度分组的贡献；不改标签或重选模型。"""
import argparse
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in ("common", "scripts"):
    sys.path.insert(0, str(ROOT / directory))
from artifacts import read_json, write_json, sha256, now
from compare import compare

VERSIONS = ("v1002", "v1101", "v1102", "v1201", "v1401")
BANDS = ("central_80pct", "next_15pct", "next_4pct", "extreme_1pct")
KEYS = ["date", "instrument"]


def components(x, y, dates):
    x, y = x.astype(np.float64), y.astype(np.float64)
    dx = x - x.groupby(dates).transform("mean")
    dy = y - y.groupby(dates).transform("mean")
    vx = (dx * dx).groupby(dates).transform("sum")
    vy = (dy * dy).groupby(dates).transform("sum")
    denominator = np.sqrt(vx * vy)
    contribution = dx * dy / denominator.where(denominator > 0)
    return contribution, dx * dx / vx.where(vx > 0), dy * dy / vy.where(vy > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/ic-contributions-20261022.json")
    args = parser.parse_args()
    summaries = {version: read_json(ROOT / f"experiments/{version}-gru.json") for version in VERSIONS}
    runs = [ROOT / summaries[v]["run_path"] for v in VERSIONS]
    protocol = read_json(runs[0] / "protocol.json")
    data = read_json(runs[0] / "data-manifest.json")
    source_hashes = {}
    for table in (protocol["universe_table"], protocol["label_table"]):
        path = Path(data["root"]) / (table + ".parquet")
        expected = next(row for row in data["files"] if row["path"] == path.name)
        source_hashes[path.name] = sha256(path)
        if source_hashes[path.name] != expected["sha256"]:
            raise ValueError("诊断数据与冻结运行来源不一致")
    result = {"computed_at": now(), "script_sha256": sha256(__file__), "versions_fixed_before_analysis": list(VERSIONS),
              "method": "全截面中心化与分母保持不变，逐样本相关性贡献按真实标签偏离截面均值的绝对值平均秩分组，再按截面等权平均。",
              "bands": list(BANDS), "percentile_cutoffs": [.8, .95, .99], "tie_method": "average",
              "label_source": "直接使用已提供的 vwap_return，按原协议处理有限性和窗口边界；未重算收益公式。",
              "source_hashes": source_hashes, "formal_metrics_modified": False, "exposure_read": False,
              "results": [], "deltas_from_v1002": [],
              "limitations": ["分组使用真实标签，只用于事后诊断，不能作为推理输入或模型切换条件。",
                              "分组贡献是完整截面相关性的加法分解，不是子样本重新计算的IC。",
                              "贡献差包含预测方差分母变化，不能单独归因为某个损失或架构组件。",
                              "验证/测试均为持续研究中已观察区间，不是新的独立盲测；不改变选模或平台晋级门槛。"]}
    for split in ("validation", "test"):
        compare(runs, split, allow_selection_metric_change=True)
        interval = protocol["splits"][split]
        filters = [("date", ">=", pd.Timestamp(interval["start"])), ("date", "<", pd.Timestamp(interval["stop"]))]
        universe = pd.read_parquet(Path(data["root"]) / (protocol["universe_table"] + ".parquet"), columns=KEYS, filters=filters)
        labels = pd.read_parquet(Path(data["root"]) / (protocol["label_table"] + ".parquet"),
                                 columns=KEYS + [protocol["label_column"]], filters=filters)
        universe = universe.merge(labels, on=KEYS, how="left", validate="one_to_one")
        horizon = universe.date + pd.Timedelta(minutes=protocol["label_horizon_minutes"])
        admitted = (horizon < pd.Timestamp(interval["stop"])) & (horizon.dt.normalize() == universe.date.dt.normalize())
        universe.loc[~admitted, protocol["label_column"]] = np.nan
        for version, run in zip(VERSIONS, runs):
            path = run / "evaluation" / split / "predictions.parquet"
            predictions = pd.read_parquet(path)
            if len(predictions) != len(universe) or not np.isfinite(predictions.factor).all():
                raise ValueError("预测覆盖或有限性发生变化")
            frame = universe.merge(predictions, on=KEYS, how="outer", validate="one_to_one", indicator=True)
            if not frame._merge.eq("both").all():
                raise ValueError("预测键与原股票池不一致")
            frame = frame.loc[np.isfinite(frame[protocol["label_column"]])].copy()
            dates, target = frame.date, frame[protocol["label_column"]].astype(np.float64)
            deviations = (target - target.groupby(dates).transform("mean")).abs()
            percentile = deviations.groupby(dates).rank(method="average", pct=True)
            band = np.searchsorted(np.array([.8, .95, .99]), percentile.to_numpy(), side="left")
            ic, factor_variance, label_variance = components(frame.factor, target, dates)
            rank_x = frame.factor.groupby(dates).rank(method="average")
            rank_y = target.groupby(dates).rank(method="average")
            rank_ic, _, _ = components(rank_x, rank_y, dates)
            count = frame.groupby("date").size()
            sections = pd.read_parquet(run / "evaluation" / split / "cross-sections.parquet").set_index("date")
            reconstructed = pd.DataFrame({"ic": ic.groupby(dates).sum(min_count=1),
                                          "rank_ic": rank_ic.groupby(dates).sum(min_count=1)})
            reconstructed.loc[count < protocol["min_cross_section_size"], :] = np.nan
            for key in ("ic", "rank_ic"):
                if not np.allclose(reconstructed[key].reindex(sections.index), sections[key], rtol=0, atol=1e-12, equal_nan=True):
                    raise ValueError("逐截面贡献之和未还原冻结指标")
            detail = pd.DataFrame({"date": dates, "band": band, "ic": ic, "rank_ic": rank_ic,
                                   "factor_variance_share": factor_variance, "label_variance_share": label_variance,
                                   "sample_fraction": 1 / dates.map(count)})
            grouped = detail.groupby(["date", "band"]).sum(min_count=1)
            index = pd.MultiIndex.from_product([sections.index, range(4)], names=["date", "band"])
            grouped = grouped.reindex(index, fill_value=0)
            for key in ("ic", "rank_ic"):
                grouped.loc[~grouped.index.get_level_values("date").isin(sections.index[sections[key].notna()]), key] = np.nan
            means = grouped.groupby("band").mean()
            frozen = read_json(run / "evaluation" / split / "metrics.json")
            for key in ("ic", "rank_ic"):
                if not np.isclose(means[key].sum(), frozen[key + "_mean"], atol=1e-12, rtol=0):
                    raise ValueError("分组均值未还原总指标")
            monthly = grouped.reset_index()
            monthly["month"] = monthly.date.dt.strftime("%Y-%m")
            monthly = monthly.groupby(["month", "band"])[means.columns].mean().reset_index()
            monthly["band"] = monthly.band.map(dict(enumerate(BANDS)))
            row = {"solution_id": version, "run_id": run.name, "split": split, "predictions_sha256": sha256(path),
                   "evaluated_rows": len(frame), "cross_sections": len(sections),
                   "ic_mean": frozen["ic_mean"], "rank_ic_mean": frozen["rank_ic_mean"],
                   "contribution_sums_match_frozen_sections": True,
                   "bands": {BANDS[i]: means.loc[i].to_dict() for i in range(4)}, "monthly": monthly.to_dict("records")}
            result["results"].append(row)
            output = ROOT / "outputs/analysis/ic-contributions-20261022" / run.name
            output.mkdir(parents=True, exist_ok=True)
            grouped.reset_index().to_parquet(output / (split + ".parquet"), index=False)
            print({"version": version, "split": split, "ic": frozen["ic_mean"], "rank_ic": frozen["rank_ic_mean"]}, flush=True)
        rows = [r for r in result["results"] if r["split"] == split]
        parent = rows[0]
        for row in rows[1:]:
            result["deltas_from_v1002"].append({"solution_id": row["solution_id"], "split": split,
                "ic_delta": row["ic_mean"] - parent["ic_mean"], "rank_ic_delta": row["rank_ic_mean"] - parent["rank_ic_mean"],
                "bands": {name: {key: row["bands"][name][key] - parent["bands"][name][key]
                                  for key in ("ic", "rank_ic", "factor_variance_share")} for name in BANDS}})
    write_json(args.output, result)


if __name__ == "__main__":
    main()
