"""复用已冻结训练统计量，顺序处理验证／测试数据；不拟合、不读取标签。"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "common"))

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from artifacts import now, read_json, sha256
from v102_preprocessing import FIELDS, FREQUENCIES, Normalizer
from v102_prepare import atomic_json, initialize, log_event, selected_files, transform_file, validate_config

TOTAL_KEYS = ["rows", "bytes", "missing_input_values", "missing_output_values", "fixed_values",
              "dynamic_values", "pooled_values", "unavailable_values", "eligible_rows",
              "cross_sections", "degenerate_cross_section_fields"]


def load_training_reference(training_run):
    training_run = Path(training_run).resolve()
    manifest = read_json(training_run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("训练数据预处理尚未完成，不能复用其参数")
    config = read_json(training_run / "config.json")
    protocol = read_json(training_run / "protocol.json")
    validate_config(config)
    for name, key in [("config.json", "config_sha256"), ("protocol.json", "protocol_sha256"),
                      ("summary.json", "summary_sha256"), ("data-manifest.json", "data_manifest_sha256")]:
        if sha256(training_run / name) != manifest[key]:
            raise ValueError(f"训练产物已改变: {name}")
    # 保持与训练数据完全相同的数值变换和时间截面对齐实现。
    for name in ["v102_preprocessing.py", "v102_prepare.py"]:
        relative = "solutions/v102/" + name
        if sha256(HERE / name) != manifest["source"]["files"][relative]:
            raise ValueError(f"预处理核心实现与训练快照不同: {name}")
    intervals = protocol["splits"]
    if (pd.Timestamp(intervals["train"]["stop"]) != pd.Timestamp(intervals["validation"]["start"])
            or pd.Timestamp(intervals["validation"]["stop"]) != pd.Timestamp(intervals["test"]["start"])
            or any(pd.Timestamp(intervals[key]["start"]) >= pd.Timestamp(intervals[key]["stop"])
                   for key in ["train", "validation", "test"])):
        raise ValueError("本入口要求训练、验证、测试区间连续且严格有序，避免丢失动态历史")
    summary = read_json(training_run / "summary.json")
    hashes = {}
    for frequency in FREQUENCIES:
        path = training_run / "normalization" / f"{frequency}.json"
        hashes[frequency] = sha256(path)
        if hashes[frequency] != summary["frequencies"][frequency]["statistics_sha256"]:
            raise ValueError(f"训练归一化参数已改变: {frequency}")
        statistics = read_json(path)
        if (statistics["fields"] != FIELDS or statistics["frequency"] != frequency
                or statistics["fit_interval"] != intervals["train"]):
            raise ValueError("训练归一化参数的字段、频率或拟合区间不匹配")
    reference = {"run_id": training_run.name, "path": str(training_run),
                 "manifest_sha256": sha256(training_run / "manifest.json"),
                 "normalization_sha256": hashes}
    return config, protocol, reference


def apply_holdout(training_run, run, data_root, resume=False):
    training_run, run, data_root = Path(training_run).resolve(), Path(run).resolve(), Path(data_root).resolve()
    if run == training_run or training_run in run.parents:
        raise ValueError("验证／测试结果须写入独立目录，不能覆盖或嵌入训练产物")
    config, protocol, reference = load_training_reference(training_run)
    manifest = initialize(run, config, protocol, resume)
    if manifest.get("training_reference", reference) != reference:
        raise ValueError("续跑所使用的训练参数来源已改变")
    manifest.update(kind="holdout_data_preprocessing", training_reference=reference)
    atomic_json(run / "manifest.json", manifest)
    normalization_directory = run / "normalization"
    normalization_directory.mkdir(exist_ok=True)
    for frequency, expected in reference["normalization_sha256"].items():
        destination = normalization_directory / f"{frequency}.json"
        if not destination.exists():
            shutil.copyfile(training_run / "normalization" / destination.name, destination)
        if sha256(destination) != expected:
            raise ValueError("验证／测试目录内的归一化参数副本已改变")

    universe_source = data_root / (protocol["universe_table"] + ".parquet")
    universe_record = {"path": universe_source.name, "bytes": universe_source.stat().st_size,
                       "sha256": sha256(universe_source)}
    training_data = read_json(training_run / "data-manifest.json")
    if universe_record != training_data["universe"]:
        raise ValueError("股票池与训练期处理使用的快照不同")
    data_manifest_path = run / "data-manifest.json"
    data_manifest = read_json(data_manifest_path) if data_manifest_path.exists() else {
        "root": str(data_root), "universe": universe_record, "files": {}}
    if data_manifest["root"] != str(data_root) or data_manifest["universe"] != universe_record:
        raise ValueError("续跑的数据目录或股票池已改变")
    atomic_json(data_manifest_path, data_manifest)
    connection = duckdb.connect()
    connection.execute("SET threads = ?", [config["processing"]["threads"]])
    connection.execute("SET memory_limit = ?", [config["processing"]["memory_limit"]])
    connection.execute("SET temp_directory = ?", [str(run / "duckdb-temp")])
    summary = {"solution_id": "v102", "training_reference": reference,
               "fit_interval": protocol["splits"]["train"], "fields": FIELDS,
               "channels_per_frequency": 2 * len(FIELDS),
               "dynamic_history": "training_moments_then_continuous_validation_and_test_bars",
               "splits": {name: {"interval": protocol["splits"][name], "frequencies": {}}
                          for name in ["validation", "test"]}}
    for frequency in FREQUENCIES:
        statistics_path = normalization_directory / f"{frequency}.json"
        statistics = read_json(statistics_path)
        statistics_sha = reference["normalization_sha256"][frequency]
        normalizer = Normalizer(statistics, config, training=False)
        # 同一对象从验证期一直延伸到测试期；有回退需求时保留所有已知历史。
        for split in ["validation", "test"]:
            interval = protocol["splits"][split]
            records, totals = [], {}
            for source in selected_files(data_root, frequency, interval):
                started = time.perf_counter()
                relative = source.relative_to(data_root).as_posix()
                identity = {"bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
                previous = data_manifest["files"].get(relative)
                if previous is not None and any(previous[key] != identity[key] for key in identity):
                    raise ValueError(f"源数据已改变: {relative}")
                if previous is None:
                    data_manifest["files"][relative] = {**identity, "sha256": sha256(source)}
                    atomic_json(data_manifest_path, data_manifest)
                source_sha = data_manifest["files"][relative]["sha256"]
                destination = run / "data" / split / frequency / source.name
                record_path = run / "records" / split / frequency / f"{source.stem}.json"
                if record_path.exists():
                    record = read_json(record_path)
                    if (record["statistics_sha256"] != statistics_sha or record["source_sha256"] != source_sha
                            or sha256(destination) != record["sha256"]):
                        raise ValueError("已有处理结果的参数、输入或内容发生变化")
                    normalizer = Normalizer(statistics, config, training=False, history_dict=record["history"])
                else:
                    record = transform_file(connection, source, universe_source, interval, config, normalizer, destination)
                    if config["data"]["session"] == "all" and record["rows"] != pq.ParquetFile(source).metadata.num_rows:
                        raise ValueError("处理行数与原始月度文件不一致")
                    record.update(split=split, month=source.stem, statistics_sha256=statistics_sha,
                                  source_sha256=source_sha, history=normalizer.history_dict())
                    atomic_json(record_path, record)
                    log_event(run, "transformed_month", split=split, frequency=frequency, month=source.stem,
                              rows=record["rows"], bytes=record["bytes"], dynamic_values=record["dynamic_values"],
                              pooled_values=record["pooled_values"], seconds=round(time.perf_counter() - started, 2))
                records.append({key: value for key, value in record.items() if key not in {"history", "columns"}})
                for name in TOTAL_KEYS:
                    totals[name] = totals.get(name, 0) + record.get(name, 0)
            summary["splits"][split]["frequencies"][frequency] = {
                "statistics_sha256": statistics_sha, **totals, "partitions": records}
            atomic_json(run / "summary.json", summary)
            log_event(run, "split_frequency_completed", split=split, frequency=frequency, **totals)
        if sha256(statistics_path) != statistics_sha or sha256(training_run / "normalization" / statistics_path.name) != statistics_sha:
            raise ValueError("处理过程中训练归一化参数发生变化")
        del normalizer, statistics
        gc.collect()
    manifest.update(status="complete", completed_at=now(), summary_sha256=sha256(run / "summary.json"),
                    data_manifest_sha256=sha256(data_manifest_path), frozen_statistics_unchanged=True)
    atomic_json(run / "manifest.json", manifest)
    log_event(run, "holdout_preprocessing_completed", summary=str(run / "summary.json"))
    connection.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=os.environ.get("JYCAPITAL_DATA_DIR", str(ROOT / "data")))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    apply_holdout(args.training_run, args.run_dir, args.data_root, args.resume)


if __name__ == "__main__":
    main()
