"""逐字段统计原始 Bar 第一层无效值，不归一化、不切窗、不训练。"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
# 固定统计口径，避免后续 v3 的清洗试验改变已记录的第一层结论。
sys.path.insert(0, str(ROOT / "solutions/v2"))
from v2_preprocessing import FIELDS, PRICE_FIELDS, FREQUENCIES, log_values

SCOPES = ("all_sessions", "regular_sessions")
CAUSES = ("null", "nonfinite_nonnull", "zero_price", "negative")


def write_field_csv(path, scopes):
    columns = ["scope", "split", "frequency", "field", "bar_rows", "invalid_values",
               "invalid_fraction", *CAUSES]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for scope, data in scopes.items():
            for split, frequencies in data["by_split_and_frequency"].items():
                for frequency, statistics in frequencies.items():
                    for field, count in statistics["fields"].items():
                        writer.writerow({"scope": scope, "split": split, "frequency": frequency,
                                         "field": field, "bar_rows": statistics["rows"],
                                         "invalid_values": count["invalid"], "invalid_fraction": count["fraction"],
                                         **{cause: count[cause] for cause in CAUSES}})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def counts(values, nulls, price):
    finite = np.isfinite(values)
    return (
        nulls,
        ~finite & ~nulls,
        finite & (values == 0) if price else np.zeros(len(values), bool),
        finite & (values < 0),
    )


def check_rules():
    frame = pd.DataFrame({field: [1., 0., -1., np.nan, np.inf, -np.inf] for field in FIELDS})
    expected = ~np.isfinite(log_values(frame))
    for index, field in enumerate(FIELDS):
        values = frame[field].to_numpy()
        actual = np.logical_or.reduce(counts(values, np.isnan(values), field in PRICE_FIELDS))
        np.testing.assert_array_equal(actual, expected[:, index])
    return {"source_log_values_parity": True, "fields": 36, "cases_per_field": 6}


def aggregate(partitions, scope, split=None, frequency=None):
    selected = [p for p in partitions if (split is None or p["split"] == split)
                and (frequency is None or p["frequency"] == frequency)]
    rows = sum(p["scopes"][scope]["rows"] for p in selected)
    fields = {}
    for field in FIELDS:
        causes = {cause: sum(p["scopes"][scope]["fields"][field][cause] for p in selected) for cause in CAUSES}
        invalid = sum(causes.values())
        fields[field] = {"invalid": invalid, "fraction": invalid / rows if rows else None, **causes}
    total = sum(field["invalid"] for field in fields.values())
    return {
        "rows": rows, "field_values": rows * len(FIELDS), "invalid_values": total,
        "invalid_value_fraction": total / (rows * len(FIELDS)) if rows else None,
        "field_names_with_invalid_values": [field for field in FIELDS if fields[field]["invalid"]],
        "bar_rows_with_any_invalid": sum(p["scopes"][scope]["bar_rows_with_any_invalid"] for p in selected),
        "bar_rows_with_non_book_invalid": sum(p["scopes"][scope]["bar_rows_with_non_book_invalid"] for p in selected),
        "bar_rows_with_book_invalid": sum(p["scopes"][scope]["bar_rows_with_book_invalid"] for p in selected),
        "fields": fields,
    }


def scan_file(path, split, frequency, run, protocol):
    source = pq.ParquetFile(path)
    total = np.zeros((2, len(FIELDS), len(CAUSES)), np.int64)
    denominators = np.zeros(2, np.int64)
    group_rows = np.zeros((2, 3), np.int64)
    interval = protocol["splits"][split]
    lo, hi = pd.Timestamp(interval["start"]).value, pd.Timestamp(interval["stop"]).value
    for group_index in range(source.num_row_groups):
        group = source.metadata.row_group(group_index)
        selected_fields = []
        for field in FIELDS:
            index = source.schema_arrow.get_field_index(field)
            statistics = group.column(index).statistics
            dtype = source.schema_arrow.field(field).type
            # 整数无非有限值；无空值、最小值非负可证明数量字段全部合法。
            if field not in PRICE_FIELDS and pa.types.is_integer(dtype) and statistics and statistics.has_min_max and statistics.null_count == 0 and statistics.min >= 0:
                continue
            selected_fields.append(field)
        table = source.read_row_group(group_index, columns=["date", *selected_fields], use_threads=True)
        dates = table["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        in_split = (dates >= lo) & (dates < hi)
        minute = (dates // 60_000_000_000) % 1440
        regular = in_split & (((minute > 570) & (minute <= 690)) | ((minute > 780) & (minute <= 900)))
        selectors = (in_split, regular)
        denominators += [int(selector.sum()) for selector in selectors]
        core_bad = np.zeros(len(dates), bool)
        book_bad = np.zeros(len(dates), bool)
        for field in selected_fields:
            values = table[field].to_numpy()
            nulls = table[field].is_null().to_numpy()
            masks = counts(values, nulls, field in PRICE_FIELDS)
            bad = np.logical_or.reduce(masks)
            index = FIELDS.index(field)
            for scope_index, selector in enumerate(selectors):
                for cause_index, mask in enumerate(masks):
                    total[scope_index, index, cause_index] += int((mask & selector).sum())
            if index < 6:
                core_bad |= bad
            else:
                book_bad |= bad
        for scope_index, selector in enumerate(selectors):
            group_rows[scope_index] += [int(((core_bad | book_bad) & selector).sum()),
                                        int((core_bad & selector).sum()), int((book_bad & selector).sum())]
    scopes = {}
    for scope_index, scope in enumerate(SCOPES):
        scopes[scope] = {
            "rows": int(denominators[scope_index]),
            "bar_rows_with_any_invalid": int(group_rows[scope_index, 0]),
            "bar_rows_with_non_book_invalid": int(group_rows[scope_index, 1]),
            "bar_rows_with_book_invalid": int(group_rows[scope_index, 2]),
            "fields": {field: {cause: int(total[scope_index, field_index, cause_index]) for cause_index, cause in enumerate(CAUSES)} for field_index, field in enumerate(FIELDS)},
        }
    preprocessing = ROOT / "outputs/preprocessing" / ("v2-20261010" if split == "train" else "v2-holdout-20261011")
    relative = Path(frequency) / (path.stem + ".json") if split == "train" else Path(split) / frequency / (path.stem + ".json")
    record = read_json(preprocessing / "records" / relative)
    invalid = int(total[0].sum())
    assert denominators[0] == record["rows"], (path, "row_count")
    assert invalid == record["missing_input_values"], (path, invalid, record["missing_input_values"])
    result = {"split": split, "frequency": frequency, "month": path.stem,
              "source": {"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns},
              "matches_frozen_preprocessing": True, "scopes": scopes}
    write_json(run / "partitions" / (frequency + "-" + path.stem + ".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "partitions").mkdir()
    pa.set_cpu_count(8)
    manifest = {"status": "running", "pid": os.getpid(), "scope": "raw_field_layer_one_only", "training_started": False}
    write_json(run / "manifest.json", manifest)
    print(json.dumps({"status": "running", "pid": os.getpid(), "run_dir": str(run)}), flush=True)
    try:
        checks = check_rules()
        protocol = read_json(ROOT / "evaluation/protocols/local-v1.json")
        partitions = []
        for period in pd.period_range("2020-01", "2024-12", freq="M"):
            month = str(period)
            stamp = period.start_time
            split = next(name for name, interval in protocol["splits"].items() if pd.Timestamp(interval["start"]) <= stamp < pd.Timestamp(interval["stop"]))
            for frequency in FREQUENCIES:
                path = args.data_root / ("cpt_jyc_2026_stock_" + frequency) / (month + ".parquet")
                partitions.append(scan_file(path, split, frequency, run, protocol))
            write_json(run / "progress.json", {"month": month, "split": split, "partitions_completed": len(partitions), "partitions_total": 240})
            with (run / "events.jsonl").open("a", encoding="utf-8") as events:
                events.write(json.dumps({"event": "month_scanned", "month": month, "partitions_completed": len(partitions)}) + "\n")
        scopes = {
            scope: {
                "combined": aggregate(partitions, scope),
                "by_frequency": {frequency: aggregate(partitions, scope, frequency=frequency) for frequency in FREQUENCIES},
                "by_split": {split: aggregate(partitions, scope, split=split) for split in protocol["splits"]},
                "by_split_and_frequency": {split: {frequency: aggregate(partitions, scope, split=split, frequency=frequency) for frequency in FREQUENCIES} for split in protocol["splits"]},
            } for scope in SCOPES
        }
        assert scopes["all_sessions"]["combined"]["rows"] == 698883381
        summary = {
            "analysis_id": run.name, "status": "complete", "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "scope": "raw_field_layer_one_only", "training_started": False,
            "rules": {"price_fields": sorted(PRICE_FIELDS), "price_valid": "finite_and_strictly_positive", "quantity_valid": "finite_and_nonnegative", "causes": list(CAUSES), "counting_unit": "one_field_value_in_one_existing_original_bar_counted_once", "stock_scope": "all_original_stocks_not_restricted_to_prediction_pool", "all_sessions": "all_original_bar_times", "regular_sessions": "(09:30,11:30] and (13:00,15:00]", "absent_bars": "not_counted_as_field_values"},
            "scopes": scopes, "partitions": partitions,
            "checks": {**checks, "partitions": 240, "all_partition_counts_match_v2_preprocessing": True, "row_count_matches_v2_preprocessing": True},
            "source_sha256": {"script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "field_rules": hashlib.sha256((ROOT / "solutions/v2/v2_preprocessing.py").read_bytes()).hexdigest()},
        }
        write_field_csv(run / "field-counts.csv", scopes)
        write_json(run / "summary.json", summary)
        manifest.update(status="complete", summary_path=str(run / "summary.json"))
        write_json(run / "manifest.json", manifest)
        print(json.dumps({"status": "complete", "summary": str(run / "summary.json"), "rows": scopes["all_sessions"]["combined"]["rows"], "invalid_values": scopes["all_sessions"]["combined"]["invalid_values"]}), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        write_json(run / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
