"""按 v2 固定历史窗口统计完整样本数量，不训练模型或更改预处理。"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions/v2"))
from v2_data import FREQUENCIES, MINUTE_NS, WindowBlock, regular_times
from v2_preprocessing import FIELDS, PRICE_FIELDS, output_columns


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_valid(values, price):
    return np.isfinite(values) & ((values > 0) if price else (values >= 0))


def window_any(flags, right, stocks, length):
    """每个样本的窗口中是否有 True；窗口不足由调用者另行检查。"""
    prefix = np.zeros((len(flags) + 1, flags.shape[1]), np.int32)
    np.cumsum(flags, axis=0, dtype=np.int32, out=prefix[1:])
    return (prefix[(right + 1).clip(0, len(flags)), stocks]
            - prefix[(right - length + 1).clip(0, len(flags)), stocks]) > 0


def classify_windows(times, present, field_bits, extra_missing, requested, stocks, length, max_age):
    right = np.searchsorted(times, requested, side="right") - 1
    left = right - length + 1
    insufficient = left < 0
    if len(times):
        too_old = (requested - times[left.clip(0, len(times) - 1)]) / MINUTE_NS > max_age
        absent = window_any(~present, right, stocks, length)
        normalized_extra = window_any(extra_missing, right, stocks, length)
    else:
        too_old = np.zeros(len(requested), bool)
        absent = np.ones(len(requested), bool)
        normalized_extra = np.zeros(len(requested), bool)
    bits = np.zeros(len(requested), np.uint64)
    if len(times):
        for start in range(0, len(requested), 8192):
            stop = min(start + 8192, len(requested))
            positions = right[start:stop, None] - np.arange(length - 1, -1, -1)
            selected = field_bits[positions.clip(0, len(times) - 1), stocks[start:stop, None]]
            bits[start:stop] = np.bitwise_or.reduce(np.where(positions >= 0, selected, np.uint64(0)), axis=1)
    raw_complete = ~(insufficient | too_old | absent | (bits != 0))
    return {
        "raw_complete": raw_complete,
        "normalized_complete": raw_complete & ~normalized_extra,
        "insufficient_history": insufficient | too_old,
        "missing_bar": absent,
        "invalid_fields": bits,
        "normalization_extra": normalized_extra,
    }


def self_check(config):
    """用实际 WindowBlock 的 72 个掩码独立核对窗口统计。"""
    assert raw_valid(np.array([0.0, -1.0, np.nan, np.inf, 1.0]), False).tolist() == [True, False, False, False, True]
    assert raw_valid(np.array([0.0, -1.0, np.nan, np.inf, 1.0]), True).tolist() == [False, False, False, False, True]
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-05"])
    keys = pd.DataFrame({"date": pd.to_datetime(["2024-01-02 09:20", "2024-01-02 14:00", "2024-01-03 14:00", "2024-02-05 09:31"]), "instrument": "A"})
    grids, expected = {}, {}
    for frequency in FREQUENCIES:
        times = regular_times(days, frequency)
        present = np.ones((len(times), 1), bool)
        bits = np.zeros((len(times), 1), np.uint64)
        extras = np.zeros((len(times), 1), bool)
        values = np.ones((len(times), 1, 72), np.float32)
        bad = np.searchsorted(times, pd.Timestamp("2024-01-02 13:30").value)
        present[bad, 0] = False
        values[bad, 0] = np.nan
        bits[bad + 1, 0] = np.uint64(1)
        values[bad + 1, 0, [0, 36]] = np.nan
        extras[bad + 2, 0] = True
        values[bad + 2, 0, 36] = np.nan
        grids[frequency] = times, values
        result = classify_windows(times, present, bits, extras, keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64), np.zeros(len(keys), int), config["data"]["window_bars"][frequency], config["data"]["history_days"] * 1440)
        expected[frequency] = result["normalized_complete"]
    windows, _ = WindowBlock(keys, grids, ["A"], config).batch(np.arange(len(keys)))
    for frequency in FREQUENCIES:
        np.testing.assert_array_equal(expected[frequency], (windows[frequency][:, 72:144] == 1).all(axis=(1, 2)))
    return {"raw_zero_and_invalid_rules": "passed", "actual_windowblock_mask_parity": "passed", "samples": len(keys), "frequencies": 4}


class Scanner:
    def __init__(self, data_root, training_run, holdout_run, source_run, run):
        self.data_root, self.training_run, self.holdout_run, self.source_run = map(Path, (data_root, training_run, holdout_run, source_run))
        self.run = Path(run)
        self.config = read_json(self.source_run / "config.json")
        self.protocol = read_json(self.source_run / "protocol.json")
        if self.config.get("solution_id") != "v2":
            raise ValueError("此统计使用 v2 的逐字段掩码和窗口，须提供 v2 运行")
        self.records, self.extra_partitions, self.source_files = {}, {}, {}
        for frequency in FREQUENCIES:
            statistics = read_json(self.training_run / "normalization" / (frequency + ".json"))
            assert min(statistics["global"]["count"]) >= 30, "训练全体字段统计量不足，不能使用缺失等价证明"
        self.universe = pd.read_parquet(self.data_root / (self.protocol["universe_table"] + ".parquet"), columns=["date", "instrument"])
        self.universe["date"] = pd.to_datetime(self.universe.date)
        self.universe["instrument"] = self.universe.instrument.astype("category")
        assert not self.universe.duplicated(["date", "instrument"]).any()
        self.days = np.sort(self.universe.date.dt.normalize().unique())
        self.checks = {"synthetic": self_check(self.config), "actual_processed_windows": []}

    def log(self, event, **values):
        with (self.run / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), "event": event, **values}, ensure_ascii=False) + "\n")

    def split_for(self, month):
        stamp = pd.Period(month, "M").start_time
        return next((name for name, interval in self.protocol["splits"].items() if pd.Timestamp(interval["start"]) <= stamp < pd.Timestamp(interval["stop"])), None)

    def processed_path(self, frequency, month):
        split = self.split_for(month)
        return (self.training_run / "data" if split == "train" else self.holdout_run / "data" / split) / frequency / (month + ".parquet")

    def quality(self, frequency, month):
        key = frequency, month
        if key not in self.records:
            split = self.split_for(month)
            root = self.training_run if split == "train" else self.holdout_run
            relative = Path(frequency) / (month + ".json") if split == "train" else Path(split) / frequency / (month + ".json")
            record = read_json(root / "records" / relative)
            assert record["unavailable_values"] == 0
            extra = record["missing_output_values"] - 2 * record["missing_input_values"]
            assert extra >= 0
            self.records[key] = record
            self.extra_partitions[frequency + "/" + month] = extra
        return self.records[key]

    def load_raw_grid(self, frequency, times, stocks):
        shape = len(times), len(stocks)
        present = np.zeros(shape, bool)
        field_bits = np.zeros(shape, np.uint64)
        extras = np.zeros(shape, bool)
        stock_set = pa.array(stocks.astype(str).to_numpy())
        if not len(times):
            return present, field_bits, extras
        lo, hi = pd.Timestamp(times[0]), pd.Timestamp(times[-1])
        for period in pd.period_range(lo, hi, freq="M"):
            month = str(period)
            if self.split_for(month) is None:
                continue
            path = self.data_root / ("cpt_jyc_2026_stock_" + frequency) / (month + ".parquet")
            source = pq.ParquetFile(path)
            self.source_files[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "rows": source.metadata.num_rows}
            schema = source.schema_arrow
            field_index = {name: schema.get_field_index(name) for name in ["date", *FIELDS]}
            for group_index in range(source.num_row_groups):
                group = source.metadata.row_group(group_index)
                time_stats = group.column(field_index["date"]).statistics
                if time_stats and time_stats.has_min_max and (pd.Timestamp(time_stats.max) < lo or pd.Timestamp(time_stats.min) > hi):
                    continue
                selected_fields = []
                for field in FIELDS:
                    stats = group.column(field_index[field]).statistics
                    # 整数列的元数据能证明全部有效；浮点列仍读取，避免遗漏 NaN。
                    if pa.types.is_integer(schema.field(field).type) and stats and stats.has_min_max and stats.null_count == 0 and stats.min >= 0:
                        continue
                    selected_fields.append(field)
                table = source.read_row_group(group_index, columns=["date", "instrument", *selected_fields], use_threads=True)
                dates = table["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
                positions = np.searchsorted(times, dates)
                stock_ids = pc.fill_null(pc.index_in(table["instrument"], value_set=stock_set), -1).to_numpy()
                take = (positions < len(times)) & (stock_ids >= 0)
                take &= times[positions.clip(0, len(times) - 1)] == dates
                if not take.any():
                    continue
                ti, si = positions[take], stock_ids[take]
                assert not present[ti, si].any(), "原始行情出现重复键"
                present[ti, si] = True
                row_bits = np.zeros(int(take.sum()), np.uint64)
                for field in selected_fields:
                    raw = table[field].to_numpy()[take]
                    bad = ~raw_valid(raw, field in PRICE_FIELDS)
                    row_bits[bad] |= np.uint64(1 << FIELDS.index(field))
                field_bits[ti, si] = row_bits
            record = self.quality(frequency, month)
            if record["missing_output_values"] > 2 * record["missing_input_values"]:
                # 若 c 比 z 多缺失，直接读取该分区掩码，不依赖等价推断。
                normalized = pq.ParquetFile(self.processed_path(frequency, month))
                for group_index in range(normalized.num_row_groups):
                    group = normalized.metadata.row_group(group_index)
                    time_stats = group.column(0).statistics
                    if time_stats and time_stats.has_min_max and (pd.Timestamp(time_stats.max) < lo or pd.Timestamp(time_stats.min) > hi):
                        continue
                    columns = [group.column(index).path_in_schema for index in range(2, group.num_columns) if not group.column(index).statistics or group.column(index).statistics.null_count != 0]
                    table = normalized.read_row_group(group_index, columns=["date", "instrument", *columns], use_threads=True)
                    dates = table["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
                    positions = np.searchsorted(times, dates)
                    stock_ids = pc.fill_null(pc.index_in(table["instrument"], value_set=stock_set), -1).to_numpy()
                    take = (positions < len(times)) & (stock_ids >= 0)
                    take &= times[positions.clip(0, len(times) - 1)] == dates
                    ti, si = positions[take], stock_ids[take]
                    bad = np.zeros(int(take.sum()), bool)
                    for column in columns:
                        bad |= ~np.isfinite(table[column].to_numpy()[take])
                    extras[ti, si] = bad & (field_bits[ti, si] == 0)
        return present, field_bits, extras

    def actual_window_check(self, keys, results, month, full_times):
        # 选取完整、字段不完整、缺 Bar、历史不足等类别，再与实际 72 掩码比较。
        candidates = []
        for frequency in FREQUENCIES:
            flags = results[frequency]
            for selector in [flags["raw_complete"], flags["invalid_fields"] != 0, flags["missing_bar"], flags["insufficient_history"]]:
                ids = np.flatnonzero(selector)
                if len(ids):
                    candidates.extend(ids[np.linspace(0, len(ids) - 1, min(3, len(ids))).astype(int)])
        selected = np.unique(candidates)
        check_keys = keys.iloc[selected].reset_index(drop=True)
        stocks = pd.Index(sorted(check_keys.instrument.unique()))
        grids = {}
        for frequency, times in full_times.items():
            needed = []
            right = np.searchsorted(times, check_keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64), side="right") - 1
            length = self.config["data"]["window_bars"][frequency]
            for endpoint in right:
                needed.extend(times[max(0, endpoint - length + 1):endpoint + 1])
            needed = np.unique(needed)
            values = np.full((len(times), len(stocks), 72), np.nan, np.float32)
            if len(needed):
                first, last = pd.Timestamp(needed[0]), pd.Timestamp(needed[-1])
                for period in pd.period_range(first, last, freq="M"):
                    if self.split_for(str(period)) is None:
                        continue
                    table = pq.read_table(self.processed_path(frequency, str(period)), columns=output_columns(), filters=[("date", "in", list(pd.to_datetime(needed))), ("instrument", "in", list(stocks))])
                    frame = table.to_pandas()
                    ti = pd.Index(times).get_indexer(frame.date.to_numpy(dtype="datetime64[ns]").astype(np.int64))
                    si = stocks.get_indexer(frame.instrument)
                    assert (ti >= 0).all() and (si >= 0).all()
                    values[ti, si] = frame[output_columns()[2:]].to_numpy(np.float32)
            grids[frequency] = times, values
        windows, _ = WindowBlock(check_keys, grids, stocks, self.config).batch(np.arange(len(selected)))
        for frequency in FREQUENCIES:
            actual = (windows[frequency][:, 72:144] == 1).all(axis=(1, 2))
            np.testing.assert_array_equal(actual, results[frequency]["normalized_complete"][selected])
        self.checks["actual_processed_windows"].append({"month": month, "samples": len(selected), "frequencies": 4, "passed": True})

    def scan_month(self, split, month):
        started = time.perf_counter()
        lo, hi = pd.Period(month, "M").start_time, (pd.Period(month, "M") + 1).start_time
        keys = self.universe.loc[(self.universe.date >= lo) & (self.universe.date < hi), ["date", "instrument"]].sort_values(["date", "instrument"]).reset_index(drop=True)
        labels = pq.read_table(self.data_root / (self.protocol["label_table"] + ".parquet"), columns=["date", "instrument", self.protocol["label_column"]], filters=[("date", ">=", lo), ("date", "<", hi)]).to_pandas()
        keys = keys.merge(labels, on=["date", "instrument"], how="left", validate="one_to_one")
        requested = keys.date.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        horizon = keys.date + pd.Timedelta(minutes=self.protocol["label_horizon_minutes"])
        admissible = (horizon < pd.Timestamp(self.protocol["splits"][split]["stop"])) & (horizon.dt.normalize() == keys.date.dt.normalize())
        finite_label = np.isfinite(keys[self.protocol["label_column"]].to_numpy(np.float64)) & admissible.to_numpy()
        stocks = pd.Index(sorted(keys.instrument.unique()))
        stock_ids = stocks.get_indexer(keys.instrument)
        history_start = lo - pd.Timedelta(days=self.config["data"]["history_days"])
        results, full_times = {}, {}
        all_fields = np.zeros(len(keys), np.uint64)
        for frequency in FREQUENCIES:
            length = self.config["data"]["window_bars"][frequency]
            all_times = regular_times(self.days[(self.days >= history_start.to_datetime64()) & (self.days < hi.to_datetime64())], frequency)
            left = np.searchsorted(all_times, lo.value)
            times = all_times[max(0, left - length):]
            present, bits, extras = self.load_raw_grid(frequency, times, stocks)
            results[frequency] = classify_windows(times, present, bits, extras, requested, stock_ids, length, self.config["data"]["history_days"] * 1440)
            all_fields |= results[frequency]["invalid_fields"]
            full_times[frequency] = times
            self.log("frequency_scanned", split=split, month=month, frequency=frequency)
        raw_complete = np.logical_and.reduce([results[f]["raw_complete"] for f in FREQUENCIES])
        normalized_complete = np.logical_and.reduce([results[f]["normalized_complete"] for f in FREQUENCIES])
        history_short = np.logical_or.reduce([results[f]["insufficient_history"] for f in FREQUENCIES])
        missing_bar = np.logical_or.reduce([results[f]["missing_bar"] for f in FREQUENCIES])
        extra = np.logical_or.reduce([results[f]["normalization_extra"] for f in FREQUENCIES])
        if month in {"2020-01", "2024-01", "2024-12"}:
            self.actual_window_check(keys, results, month, full_times)
        def counts(selector):
            return {"all_keys": int(selector.sum()), "finite_label": int((selector & finite_label).sum())}
        sections = pd.DataFrame({"date": keys.date, "kept": raw_complete, "normalized_kept": normalized_complete, "finite_label": finite_label})
        cross = sections.groupby("date", sort=True).agg(total=("kept", "size"), kept=("kept", "sum"), normalized_kept=("normalized_kept", "sum"))
        cross["coverage"] = cross.kept / cross.total
        details = keys[["date", "instrument"]].copy()
        details["finite_label"] = finite_label
        details["raw_complete"] = raw_complete
        details["normalized_complete"] = normalized_complete
        details["insufficient_history"] = history_short
        details["missing_bar"] = missing_bar
        details["invalid_field_bits"] = all_fields
        for frequency in FREQUENCIES:
            details[frequency + "_complete"] = results[frequency]["raw_complete"]
        details.to_parquet(self.run / "eligibility" / (month + ".parquet"), index=False)
        cross.to_csv(self.run / "cross-sections" / (month + ".csv"))
        row = {
            "split": split, "month": month,
            "total": counts(np.ones(len(keys), bool)),
            "raw_kept": counts(raw_complete), "raw_discarded": counts(~raw_complete),
            "normalized_kept": counts(normalized_complete), "normalized_discarded": counts(~normalized_complete),
            "raw_kept_but_normalized_discarded": counts(raw_complete & ~normalized_complete),
            "reasons_overlapping": {"insufficient_history": counts(history_short), "missing_bar": counts(missing_bar), "invalid_fields": counts(all_fields != 0), "normalization_extra": counts(extra)},
            "reasons_exclusive": {"insufficient_history": counts(history_short), "missing_bar_after_history": counts(~history_short & missing_bar), "invalid_fields_after_history_and_presence": counts(~history_short & ~missing_bar & (all_fields != 0))},
            "per_frequency_discarded": {f: counts(~results[f]["raw_complete"]) for f in FREQUENCIES},
            "per_field_affected": {field: counts((all_fields & np.uint64(1 << index)) != 0) for index, field in enumerate(FIELDS)},
            "cross_sections": {"total": len(cross), "min_coverage": float(cross.coverage.min()), "mean_coverage": float(cross.coverage.mean()), "missing_over_40_percent": int((cross.coverage < .6).sum()), "zero_coverage": int((cross.kept == 0).sum())},
        }
        write_json(self.run / "months" / (month + ".json"), row)
        write_json(self.run / "progress.json", {"at": now(), "split": split, "month": month, "total": row["total"], "discarded": row["raw_discarded"], "month_seconds": round(time.perf_counter() - started, 2)})
        self.log("month_scanned", split=split, month=month, total=row["total"], discarded=row["raw_discarded"], seconds=round(time.perf_counter() - started, 2))
        return row


def aggregate(rows):
    def combine(items):
        return {key: sum(item[key] for item in items) for key in ("all_keys", "finite_label")}
    result = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            continue
        summary = {key: combine([row[key] for row in selected]) for key in ["total", "raw_kept", "raw_discarded", "normalized_kept", "normalized_discarded", "raw_kept_but_normalized_discarded"]}
        for group in ["reasons_overlapping", "reasons_exclusive", "per_frequency_discarded", "per_field_affected"]:
            summary[group] = {key: combine([row[group][key] for row in selected]) for key in selected[0][group]}
        summary["cross_sections"] = {
            "total": sum(row["cross_sections"]["total"] for row in selected),
            "min_coverage": min(row["cross_sections"]["min_coverage"] for row in selected),
            "missing_over_40_percent": sum(row["cross_sections"]["missing_over_40_percent"] for row in selected),
            "zero_coverage": sum(row["cross_sections"]["zero_coverage"] for row in selected),
        }
        summary["discard_fraction"] = {key: summary["raw_discarded"][key] / summary["total"][key] for key in ("all_keys", "finite_label")}
        summary["remaining_coverage"] = summary["raw_kept"]["all_keys"] / summary["total"]["all_keys"]
        summary["months"] = len(selected)
        result[split] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--source-run", type=Path, default=ROOT / "outputs/runs/v2-20261011-s42")
    parser.add_argument("--training-run", type=Path, default=ROOT / "outputs/preprocessing/v2-20261010")
    parser.add_argument("--holdout-run", type=Path, default=ROOT / "outputs/preprocessing/v2-holdout-20261011")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.exists():
        parser.error("统计运行目录已存在，请使用新目录")
    run.mkdir(parents=True, exist_ok=False)
    for name in ["months", "eligibility", "cross-sections", "source"]:
        (run / name).mkdir(exist_ok=True)
    pa.set_cpu_count(8)
    manifest = {"analysis_id": run.name, "status": "running", "started_at": now(), "pid": os.getpid(), "source_run_id": args.source_run.name, "script_sha256": sha256(__file__), "scope": "count_only_no_training"}
    write_json(run / "manifest.json", manifest)
    shutil.copy2(__file__, run / "source" / Path(__file__).name)
    print(json.dumps({"status": "running", "pid": os.getpid(), "run_dir": str(run)}), flush=True)
    try:
        scanner = Scanner(args.data_root, args.training_run, args.holdout_run, args.source_run, run)
        scanner.log("analysis_started")
        rows = []
        for split, interval in scanner.protocol["splits"].items():
            for period in pd.period_range(interval["start"], pd.Timestamp(interval["stop"]) - pd.Timedelta(1, "ns"), freq="M"):
                month = str(period)
                rows.append(scanner.scan_month(split, month))
                write_json(run / "partial-summary.json", aggregate(rows))
        result = {
            "analysis_id": run.name, "source_run_id": args.source_run.name, "checked_at": now(), "status": "complete",
            "scope": "full_windows_all_36_raw_fields_and_all_72_normalized_values",
            "rules": {"window_bars": scanner.config["data"]["window_bars"], "history_days": scanner.config["data"]["history_days"], "price_valid": "finite_and_strictly_positive", "quantity_valid": "finite_and_nonnegative", "missing_bar": "discard_sample_preserve_regular_grid", "denominators": "all_universe_keys_and_finite_admissible_label_keys", "normalization": "original_frozen_v2_parameters_and_stock_pool_not_refitted"},
            "splits": aggregate(rows), "months": rows, "checks": scanner.checks,
            "normalization_extra_missing_values_by_partition": scanner.extra_partitions,
            "normalization_equivalence": "当 unavailable_values=0 且 missing_output_values=2*missing_input_values 时，z/c 缺失与原始非法位置一致；额外缺失分区直接扫描原 72 个通道。",
            "source": {"model_config_sha256": sha256(args.source_run / "config.json"), "protocol_sha256": sha256(args.source_run / "protocol.json"), "data_manifest_sha256": sha256(args.source_run / "data-manifest.json"), "script_sha256": sha256(__file__)},
            "source_files": scanner.source_files,
        }
        expected = {"train": (7760000, 7742958), "validation": (936000, 935224), "test": (1000000, 998539)}
        for split, (all_keys, finite) in expected.items():
            assert result["splits"][split]["total"] == {"all_keys": all_keys, "finite_label": finite}
        result["checks"]["sample_denominators_match_v2_run"] = True
        write_json(run / "summary.json", result)
        manifest.update(status="complete", completed_at=now(), summary_path=str(run / "summary.json"))
        write_json(run / "manifest.json", manifest)
        scanner.log("analysis_completed")
        print(json.dumps({"status": "complete", "summary": str(run / "summary.json"), "splits": {key: {"total": value["total"], "discarded": value["raw_discarded"], "kept": value["raw_kept"]} for key, value in result["splits"].items()}}, ensure_ascii=False), flush=True)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), stopped_at=now())
        write_json(run / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
