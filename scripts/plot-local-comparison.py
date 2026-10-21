"""从冻结实验摘要生成本地 IC、Rank IC 与 IC IR 对照图。"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


def read_rows(paths):
    rows, identities, versions = [], set(), set()
    protocol_path = Path(__file__).resolve().parents[1] / "evaluation/protocols/local-v2-ic.json"
    protocol_text = protocol_path.read_text(encoding="utf-8")
    expected_protocols = {hashlib.sha256(text.encode("utf-8")).hexdigest()
                          for text in (protocol_text, protocol_text.replace("\n", "\r\n"))}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data["solution_id"]
        if not re.fullmatch(r"v[0-9]+[a-z]?", version) or version in versions:
            raise ValueError("版本标识非法或重复")
        if data["status"] != "complete" or data["selection"]["selection_metric"] != "ic_mean":
            raise ValueError("本图仅接受已完成且按验证 IC 选模的摘要")
        if data["protocol_sha256"] not in expected_protocols or not data["run_id"].endswith("-s42"):
            raise ValueError("当前图注限定为仓库 local-v2-ic 协议和 s42 运行")
        versions.add(version)
        identities.add((data["protocol_sha256"], data["data_manifest_sha256"]))
        metrics = {}
        for split in ("validation", "test"):
            item = data["metrics"][split]
            if item["coverage"] != 1 or item["min_cross_section_coverage"] != 1 or \
                    item["ic_undefined_cross_sections"] or item["rank_ic_undefined_cross_sections"]:
                raise ValueError("不能把覆盖不全或未定义截面混入当前对照图")
            metrics[split] = {key: item[key] for key in ("ic_mean", "rank_ic_mean", "ic_ir")}
            if any(not math.isfinite(value) for value in metrics[split].values()):
                raise ValueError("指标必须为有限数")
        rows.append({
            "solution_id": version, "run_id": data["run_id"],
            "model_sha256": data["selection"]["model_sha256"],
            "summary_file": path.as_posix(),
            "summary_content_sha256": hashlib.sha256(json.dumps(
                data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")).hexdigest(),
            "parameters": data["resources"]["parameters"],
            "epochs_completed": data["selection"]["epochs_completed"],
            "metrics": metrics,
        })
    if len(identities) != 1:
        raise ValueError("当前对照要求相同协议及数据指纹")
    return sorted(rows, key=lambda row: row["metrics"]["validation"]["ic_mean"], reverse=True)


def render(rows, output, font_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont, fontManager
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    if font_path is None:
        candidates = [Path("C:/Windows/Fonts/msyh.ttc"),
                      Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
        font_path = next((path for path in candidates if path.is_file()), None)
    if font_path:
        fontManager.addfont(str(font_path))
        font = FontProperties(fname=str(font_path))
    else:
        font = None
        for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei"):
            try:
                font = FontProperties(fname=findfont(name, fallback_to_default=False))
                break
            except ValueError:
                continue
        if font is None:
            raise ValueError("未找到中文字体，请用 --font 指定")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(16, max(5.8, len(rows) * 0.53 + 1.7)), sharey=True)
    colors = {"validation": "#1766a4", "test": "#ce7624"}
    labels = {"validation": "验证", "test": "测试"}
    columns = (("ic_mean", "IC"), ("rank_ic_mean", "Rank IC"), ("ic_ir", "IC IR"))
    for axis, (key, title) in zip(axes, columns):
        values = [row["metrics"][split][key] for row in rows for split in colors]
        span = max(values) - min(values)
        margin = max(span * 0.30, 0.002 if key != "ic_ir" else 0.02)
        axis.set_xlim(min(values) - margin, max(values) + margin)
        axis.axhspan(-0.42, 0.42, color="#e6f2e9", zorder=0)
        for index, row in enumerate(rows):
            pairs = sorted((row["metrics"][split][key], split) for split in colors)
            axis.plot([pair[0] for pair in pairs], [index, index], color="#b6bec5", linewidth=1.7, zorder=1)
            for side, (value, split) in enumerate(pairs):
                axis.scatter(value, index, s=49, color=colors[split], zorder=3)
                axis.annotate(f"{value:.4f}", (value, index),
                              xytext=(-9 if side == 0 else 9, 0), textcoords="offset points",
                              ha="right" if side == 0 else "left", va="center",
                              fontsize=9.5, color=colors[split])
        axis.set_title(title, fontweight="bold", pad=14)
        axis.set_yticks(range(len(rows)), [row["solution_id"] for row in rows])
        axis.tick_params(axis="y", length=0, pad=8)
        axis.grid(axis="x", color="#e1e5e9", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f" if key != "ic_ir" else "%.2f"))
    axes[0].set_ylim(len(rows) - 0.55, -0.65)
    handles = [plt.Line2D([], [], color=colors[split], marker="o", linestyle="", markersize=7,
                         label=labels[split]) for split in colors]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.96), ncol=2, frameon=False)
    figure.suptitle("近期已完成模型的本地对比", x=0.065, ha="left", fontsize=17, fontweight="bold")
    figure.text(0.065, 0.915, "按验证 IC 排序；绿色行是所列模型中的验证 IC 最优。", color="#52616b", fontsize=10)
    figure.text(0.065, 0.045,
                "2024 上半年验证、下半年测试；单种子 42。测试期已用于持续研究观察。",
                fontsize=10, color="#52616b")
    figure.text(0.065, 0.013,
                "仅展示本地指标；不包含平台 BARRA 处理、交易成本或官方综合评分。完整指标见对应冻结摘要。",
                fontsize=9.5, color="#52616b")
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.83, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.summaries)
    render(rows, args.output, args.font)
    metadata = {"generated_at": datetime.now().astimezone().isoformat(),
                "scope": "固定本地协议下的已完成单种子运行；不代表平台成绩或统计显著性。",
                "summary_digest_format": "sha256-json-sort-keys-utf8-compact",
                "rows": rows,
                "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"image": args.output.as_posix(), "models": len(rows),
                      "best_validation": rows[0]["solution_id"]}))


if __name__ == "__main__":
    main()
