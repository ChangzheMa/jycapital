"""冻结 v1601 的训练样本局部时间敏感度；不训练、不读取标签、不改变缓存。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in ("common", "solutions/v1601"):
    sys.path.insert(0, str(ROOT / directory))
from artifacts import now, read_json, sha256, write_json
from geometry_resident import DAILY_BARS, FREQUENCIES, PREVIOUS_DAYS, ResidentBlock, prepare_device_windows
from v1601_model import MultiFrequencyGRU

CLOCKS = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")
RUN_ID = "v1601-price-geometry-s42"
CACHE_KEY = "c760f3882029e185c69b39c92509beeeea3995e2214360a6196046d70158563b"


def time_profile(gradient, finite, cutoff, daily_bars):
    """只计真实有限数值通道；逐频率内归一化，避免混淆尺度。"""
    if gradient.shape != finite.shape or gradient.ndim != 2 or gradient.shape[0] != 72:
        raise ValueError("梯度及有限性须为 [72, 时间]")
    if not 0 < cutoff <= gradient.shape[1] or gradient.shape[1] % daily_bars:
        raise ValueError("截点或交易日长度非法")
    mass = np.where(finite, np.abs(gradient.astype(np.float64)), 0).sum(axis=0)
    if not np.isfinite(mass).all() or np.any(mass[cutoff:] != 0):
        raise ValueError("非有限梯度或未来敏感度")
    total = float(mass.sum())
    if total == 0:
        return None
    normalized = mass / total
    age_bars = cutoff - 1 - np.arange(len(mass))
    minutes_per_bar = 240 // daily_bars
    age_minutes = age_bars * minutes_per_bar
    bands = [("0_5", 0, 5), ("5_30", 5, 30), ("30_60", 30, 60),
             ("60_120", 60, 120), ("120_240", 120, 240), ("240_plus", 240, None)]
    by_age = {name: float(normalized[(age_minutes >= start) & (age_minutes < stop if stop else True)].sum())
              for name, start, stop in bands}
    day_index = np.arange(len(mass)) // daily_bars
    current_day = len(mass) // daily_bars - 1
    by_day = {str(offset): float(normalized[day_index == current_day - offset].sum())
              for offset in range(current_day + 1)}
    ordered = normalized[:cutoff][::-1].cumsum()
    return {"total_absolute_gradient": total, "by_trading_minute_age": by_age,
            "by_day_offset": by_day,
            "bars_covering_50_percent": int(np.searchsorted(ordered, .5)) + 1,
            "bars_covering_90_percent": int(np.searchsorted(ordered, .9)) + 1}


def read_only_block(directory):
    """只映射输入／索引，不调用常驻构造器，不打开 targets.npy。"""
    block = ResidentBlock.__new__(ResidentBlock)
    block.manifest = read_json(directory / "manifest.json")
    if block.manifest["status"] != "complete" or block.manifest["cache_key"] != CACHE_KEY:
        raise ValueError("需要冻结的 v1601 完整输入缓存")
    names = ["days", "sample_day", "sample_stock", "sample_clock"]
    names += [frequency + suffix for frequency in FREQUENCIES
              for suffix in ("-map", "-values", "-exception-pointers", "-exception-offsets", "-exception-values")]
    block.arrays = {}
    for name in names:
        array = np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        record = block.manifest["files"][name + ".npy"]
        if list(array.shape) != record["shape"] or array.dtype != np.dtype(record["dtype"]):
            raise ValueError("缓存数组元数据不符: " + name)
        block.arrays[name] = array
    block.clock = block.arrays["sample_clock"]
    block.day, block.stock = block.arrays["sample_day"], block.arrays["sample_stock"]
    block.day_plans = {f: np.arange(len(block.arrays["days"]))[:, None] + np.arange(-PREVIOUS_DAYS[f], 1)
                       for f in FREQUENCIES}
    return block


def summarize(rows):
    result = []
    for frequency in FREQUENCIES:
        for clock in CLOCKS:
            selected = [r["profile"] for r in rows if r["frequency"] == frequency and r["clock"] == clock]
            usable = [r for r in selected if r is not None]
            item = {"frequency": frequency, "clock": clock, "samples": len(selected),
                    "nonzero_gradient_samples": len(usable)}
            if usable:
                for group in ("by_trading_minute_age", "by_day_offset"):
                    item[group] = {key: float(np.mean([r[group][key] for r in usable])) for key in usable[0][group]}
                for key in ("bars_covering_50_percent", "bars_covering_90_percent"):
                    item[key + "_median"] = float(np.median([r[key] for r in usable]))
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/temporal-sensitivity-20261023.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    run = ROOT / "outputs/runs" / RUN_ID
    config, selection = read_json(run / "config.json"), read_json(run / "selection.json")
    if sha256(run / "best.pt") != selection["model_sha256"]:
        raise ValueError("冻结权重指纹不符")
    source = read_json(run / "manifest.json")["source"]
    for name in ("solutions/v1601/v1601_model.py", "solutions/v1601/v1601_schema.py", "common/geometry_resident.py"):
        if sha256(ROOT / name) != source["files"][name]:
            raise ValueError("冻结源码不符: " + name)
    model = MultiFrequencyGRU(config).eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=False)["state_dict"], strict=True)
    model.requires_grad_(False)
    cache = ROOT / "outputs/preprocessing/resident-cache" / CACHE_KEY
    block = read_only_block(cache)
    days = block.arrays["days"].astype("datetime64[ns]").astype("datetime64[D]").astype(str)
    chosen, records = [], []
    for year in range(2020, 2024):
        day = int(np.flatnonzero(np.char.startswith(days, f"{year}-06-"))[4])
        choices = np.flatnonzero((block.day == day) & (block.clock == 7))
        indices = np.random.default_rng(np.random.SeedSequence([20261023, year])).choice(choices, size=8, replace=False)
        chosen.extend(int(i) for i in sorted(indices))
    sample_fingerprints = []
    for sample_number, index in enumerate(chosen):
        raw, repair, latest = block.raw_batch([index])
        tensors = {f: torch.from_numpy(x) for f, x in raw.items()}
        repairs = {f: tuple(torch.from_numpy(x) for x in value) for f, value in repair.items()}
        finite, digests = {}, {}
        for frequency in FREQUENCIES:
            restored = tensors[frequency].float().clone()
            positions, values = repairs[frequency]
            if positions.numel():
                restored.reshape(-1).index_copy_(0, positions, values)
            finite[frequency] = torch.isfinite(restored)[0].T.numpy()
            digests[frequency] = hashlib.sha256(restored.numpy().tobytes()).hexdigest()
        windows = {f: x.detach().requires_grad_() for f, x in prepare_device_windows(tensors, repairs, torch.from_numpy(latest)).items()}
        outputs = model.forward_group(windows, torch.from_numpy(latest))[0, :, 0]
        if not torch.isfinite(outputs).all():
            raise ValueError("冻结预测非有限")
        for clock_index, clock in enumerate(CLOCKS):
            gradients = torch.autograd.grad(outputs[clock_index], tuple(windows.values()), retain_graph=clock_index < 7)
            for frequency, gradient in zip(windows, gradients):
                cutoff = PREVIOUS_DAYS[frequency] * DAILY_BARS[frequency] + clock_index * (DAILY_BARS[frequency] // 8)
                profile = time_profile(gradient[0, :72].numpy(), finite[frequency], cutoff, DAILY_BARS[frequency])
                records.append({"sample": sample_number, "year": int(days[block.day[index]][:4]),
                                "clock": clock, "frequency": frequency, "profile": profile})
        sample_fingerprints.append({"sample": sample_number, "day": days[block.day[index]],
                                    "stock_index": int(block.stock[index]), "input_sha256": digests})
        print(json.dumps({"completed_stock_days": sample_number + 1, "total_stock_days": len(chosen)}), flush=True)
    report = {"computed_at": now(), "script_sha256": sha256(__file__), "run_id": RUN_ID,
        "model_sha256": selection["model_sha256"], "cache_key": CACHE_KEY,
        "cache_manifest_sha256": sha256(cache / "manifest.json"),
        "source_files": {name: source["files"][name] for name in
                         ("solutions/v1601/v1601_model.py", "solutions/v1601/v1601_schema.py", "common/geometry_resident.py")},
        "sampling": "2020—2023每年六月第五个交易日，各随机8个14:30股票日；SeedSequence([20261023,year])，固定后对八时点求导。",
        "stock_days": len(chosen), "outputs": len(chosen) * 8,
        "model_training_performed": False, "label_table_read": False, "target_cache_read": False,
        "validation_or_test_data_read": False, "exposure_read": False,
        "input_cache_mode": "read_only_mmap_sampled_gather",
        "full_input_cache_rehashed": False, "sample_fingerprints": sample_fingerprints,
        "method": "在冻结EMA的真实训练输入处，对每个预测时点求输入局部梯度。仅累加原本有限的72个数值通道绝对梯度，各样本、频率、时点独立归一化，再对样本等权平均。缺失标志梯度不计。",
        "results": summarize(records), "year_results": {str(year): summarize([r for r in records if r["year"] == year]) for year in range(2020, 2024)},
        "limitations": ["局部梯度衡量当前模型敏感度，不是时段可预测性、因果贡献或IC增益。",
                        "各频率输入量纲不同，不比较跨频率绝对梯度。饱和和非线性可能使局部梯度低估有限扰动作用。",
                        "交易分钟年龄按规则Bar槽位累计，跳过闭市时间；09:30最近可见位置为前日收盘。",
                        "32个训练股票日是固定小样本，没有读取标签或重新评测正式IC；不据此宣称已找到0.25突破路径。",
                        "缓存来源采用已冻结清单并核对形状dtype，记录实际样本哈希；本次未重算整套缓存文件哈希。"]}
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "stock_days": len(chosen), "profiles": len(records)}))


if __name__ == "__main__":
    main()
