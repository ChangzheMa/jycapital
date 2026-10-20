"""拟合完整训练期统计量并生成三频 z/c 序列；无切窗或模型训练。"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "common"))

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from artifacts import now, read_json, sha256, snapshot_sources
from v501_preprocessing import (FIELDS, PRICE_FIELDS, FREQUENCIES, KEYS, Normalizer,
                              UniverseIndex, empty_moments, merge_moments,
                              output_columns, pack_moments, unpack_moments)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def log_event(run, event, **kwargs):
    value = {"at": now(), "event": event, **kwargs}
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def validate_config(config):
    if config["solution_id"] != "v501" or config["data"]["frequencies"] != FREQUENCIES:
        raise ValueError("v501 需要 1、5、30 分钟三种固定逻辑频率")
    if config["data"]["fields"] != FIELDS:
        raise ValueError("字段及顺序须为 v501 已确定的 36 字段")
    spec = config["normalization"]
    if not isinstance(spec["min_observations"], int) or spec["min_observations"] < 2:
        raise ValueError("min_observations 必须先确定且至少为 2")
    if spec["min_cross_section_observations"] < 2 or spec["scale_epsilon"] <= 0:
        raise ValueError("截面最少样本数或尺度下限非法")
    if config["data"]["session"] not in {"regular", "all"}:
        raise ValueError("session 尚未确定")
    if spec["fallback"] != "expanding_then_pooled" or spec["ddof"] != 0:
        raise ValueError("回退策略和总体标准差定义不匹配")


def sql_log_source(config):
    fields = config["data"]["fields"]
    columns = []
    for field in fields:
        comparison = "> 0" if field in PRICE_FIELDS else ">= 0"
        expression = f'ln(CAST("{field}" AS DOUBLE))' if field in PRICE_FIELDS else f'ln(1.0 + CAST("{field}" AS DOUBLE))'
        columns.append(f'CASE WHEN isfinite("{field}") AND "{field}" {comparison} THEN {expression} ELSE NULL END AS "{field}"')
    session = ""
    if config["data"]["session"] == "regular":
        session = " AND ((CAST(date AS TIME) > TIME '09:30:00' AND CAST(date AS TIME) <= TIME '11:30:00') OR (CAST(date AS TIME) > TIME '13:00:00' AND CAST(date AS TIME) <= TIME '15:00:00'))"
    return ("SELECT date, instrument, " + ", ".join(columns)
            + " FROM read_parquet(?) WHERE date >= CAST(? AS TIMESTAMP) AND date < CAST(? AS TIMESTAMP)" + session)


def fit_file(connection, source, interval, config):
    fields = config["data"]["fields"]
    aggregates = []
    for field in fields:
        aggregates.extend([f'count("{field}") AS "n_{field}"',
                           f'avg("{field}") AS "mean_{field}"',
                           f'var_pop("{field}") AS "var_{field}"'])
    query = "WITH transformed AS (" + sql_log_source(config) + ") SELECT instrument, count(*) AS rows, "
    query += ", ".join(aggregates) + " FROM transformed GROUP BY instrument ORDER BY instrument"
    frame = connection.execute(query, [str(source), interval["start"], interval["stop"]]).fetchdf()
    stocks = {}
    for _, row in frame.iterrows():
        count = np.array([row[f"n_{f}"] for f in fields], dtype=np.int64)
        mean = np.nan_to_num(np.array([row[f"mean_{f}"] for f in fields], dtype=np.float64), nan=0.0)
        variance = np.nan_to_num(np.array([row[f"var_{f}"] for f in fields], dtype=np.float64), nan=0.0)
        stocks[str(row["instrument"])] = pack_moments({"count": count, "mean": mean, "m2": np.maximum(variance, 0.0) * count})
    return {"rows": int(frame["rows"].sum()), "stocks": stocks}


def complete_sections(reader):
    """Arrow batch 边界可能切断一个时间截面，保留尾截面到下一块。"""
    pending = None
    for batch in reader:
        frame = batch.to_pandas()
        if pending is not None:
            frame = pd.concat([pending, frame], ignore_index=True)
        if frame.empty:
            continue
        timestamps = frame["date"].to_numpy()
        boundary = np.searchsorted(timestamps, timestamps[-1], side="left")
        pending = frame.iloc[boundary:].copy()
        if boundary:
            yield frame.iloc[:boundary].reset_index(drop=True)
    if pending is not None and len(pending):
        yield pending.reset_index(drop=True)


def load_universe(connection, source, start, stop):
    query = """WITH last_snapshot AS (
        SELECT max(date) AS date FROM read_parquet(?) WHERE date < CAST(? AS TIMESTAMP)
    ) SELECT date, instrument FROM read_parquet(?)
    WHERE date >= coalesce((SELECT date FROM last_snapshot), CAST(? AS TIMESTAMP))
      AND date < CAST(? AS TIMESTAMP) ORDER BY date, instrument"""
    frame = connection.execute(query, [str(source), str(start), str(source), str(start), str(stop)]).fetchdf()
    return UniverseIndex(frame)


def transform_file(connection, source, universe_source, interval, config, normalizer, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    # 在打开流式行情查询之前读取股票池，避免同一连接的新查询关闭旧 reader。
    month = pd.Timestamp(source.stem + "-01")
    lower = max(month, pd.Timestamp(interval["start"]))
    upper = min((month.to_period("M") + 1).start_time, pd.Timestamp(interval["stop"]))
    universe = load_universe(connection, universe_source, lower, upper)
    query = sql_log_source(config) + " ORDER BY date, instrument"
    reader = connection.execute(query, [str(source), interval["start"], interval["stop"]]).to_arrow_reader(config["processing"]["batch_rows"])
    columns = output_columns(config["data"]["fields"])
    schema = pa.schema([pa.field("date", pa.timestamp("ns")), pa.field("instrument", pa.string())]
                       + [pa.field(name, pa.float32()) for name in columns[2:]])
    totals = {}
    first_date = last_date = None
    with pq.ParquetWriter(temporary, schema, compression="zstd", compression_level=3,
                          use_dictionary=["instrument"], write_statistics=True) as writer:
        for frame in complete_sections(reader):
            result, quality = normalizer.transform(frame, universe.eligible(frame), logged=True)
            table = pa.Table.from_pandas(result, schema=schema, preserve_index=False)
            writer.write_table(table, row_group_size=config["processing"]["batch_rows"])
            for key, value in quality.items():
                totals[key] = totals.get(key, 0) + value
            if first_date is None:
                first_date = str(result["date"].iloc[0])
            last_date = str(result["date"].iloc[-1])
    os.replace(temporary, destination)
    metadata = pq.ParquetFile(destination).metadata
    if metadata.num_rows != totals.get("rows", 0):
        raise ValueError("输出 Parquet 行数与处理计数不一致")
    return {**totals, "first_date": first_date, "last_date": last_date,
            "bytes": destination.stat().st_size, "sha256": sha256(destination), "columns": columns}


def selected_files(root, frequency, interval):
    start, stop = pd.Timestamp(interval["start"]), pd.Timestamp(interval["stop"])
    directory = root / f"cpt_jyc_2026_stock_{frequency}"
    files = []
    for path in sorted(directory.glob("*.parquet")):
        if not re.fullmatch(r"\d{4}-\d{2}", path.stem):
            raise ValueError(f"无法识别月份文件名: {path.name}")
        month = pd.Timestamp(path.stem + "-01")
        if month < stop and (month.to_period("M") + 1).start_time > start:
            files.append(path)
    expected = set(pd.period_range(start, stop - pd.Timedelta(nanoseconds=1), freq="M").astype(str))
    if {p.stem for p in files} != expected:
        raise ValueError(f"{frequency} 的训练月份文件不完整")
    return files


def initialize(run, config, protocol, resume):
    if run.exists():
        if not resume:
            raise ValueError("运行目录已存在；续跑需显式 --resume")
        manifest = read_json(run / "manifest.json")
        if read_json(run / "config.json") != config or read_json(run / "protocol.json") != protocol:
            raise ValueError("续跑配置或协议已改变")
        for name, digest in manifest["source"]["files"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"续跑源码已改变: {name}")
        return manifest
    run.mkdir(parents=True)
    atomic_json(run / "config.json", config)
    atomic_json(run / "protocol.json", protocol)
    source = snapshot_sources(ROOT, run, solution_id="v501")
    manifest = {"solution_id": "v501", "run_id": run.name, "kind": "data_preprocessing",
                "status": "preparing", "created_at": now(), "source": source,
                "config_sha256": sha256(run / "config.json"), "protocol_sha256": sha256(run / "protocol.json"),
                "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                                "pyarrow": pa.__version__, "duckdb": duckdb.__version__}}
    atomic_json(run / "manifest.json", manifest)
    return manifest


def prepare(run, data_root, config, protocol, resume=False):
    validate_config(config)
    run, data_root = Path(run).resolve(), Path(data_root).resolve()
    manifest = initialize(run, config, protocol, resume)
    interval = protocol["splits"]["train"]
    if pd.Timestamp(interval["start"]) >= pd.Timestamp(interval["stop"]):
        raise ValueError("训练区间为空")
    connection = duckdb.connect()
    connection.execute("SET threads = ?", [config["processing"]["threads"]])
    connection.execute("SET memory_limit = ?", [config["processing"]["memory_limit"]])
    connection.execute("SET temp_directory = ?", [str(run / "duckdb-temp")])
    universe_source = data_root / (protocol["universe_table"] + ".parquet")
    source_record = {"path": universe_source.name, "bytes": universe_source.stat().st_size, "sha256": sha256(universe_source)}
    data_manifest_path = run / "data-manifest.json"
    data_manifest = read_json(data_manifest_path) if data_manifest_path.exists() else {"root": str(data_root), "universe": source_record, "files": {}}
    if data_manifest["root"] != str(data_root) or data_manifest["universe"] != source_record:
        raise ValueError("数据根目录或股票池发生变化")
    atomic_json(data_manifest_path, data_manifest)
    summary = {"solution_id": "v501", "fit_interval": interval, "fields": FIELDS,
               "channels_per_frequency": 2 * len(FIELDS), "frequencies": {},
               "model_status": "not_designed", "window_status": "not_selected"}
    for frequency in FREQUENCIES:
        files = selected_files(data_root, frequency, interval)
        stock_moments, fit_rows = {}, 0
        for source in files:
            started = time.perf_counter()
            relative = source.relative_to(data_root).as_posix()
            identity = {"bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
            previous = data_manifest["files"].get(relative)
            if previous is not None and any(previous[key] != identity[key] for key in identity):
                raise ValueError(f"源数据已改变: {relative}")
            if previous is None:
                data_manifest["files"][relative] = {**identity, "sha256": sha256(source)}
                atomic_json(data_manifest_path, data_manifest)
            partial_path = run / "fit-partials" / frequency / f"{source.stem}.json"
            if partial_path.exists():
                partial = read_json(partial_path)
                if partial["source_sha256"] != data_manifest["files"][relative]["sha256"]:
                    raise ValueError("归一化中间统计来源不一致")
            else:
                partial = fit_file(connection, source, interval, config)
                partial["source_sha256"] = data_manifest["files"][relative]["sha256"]
                atomic_json(partial_path, partial)
                log_event(run, "fit_month", frequency=frequency, month=source.stem, rows=partial["rows"], seconds=round(time.perf_counter() - started, 2))
            fit_rows += partial["rows"]
            for stock, value in partial["stocks"].items():
                stock_moments[stock] = merge_moments(stock_moments.get(stock, empty_moments(len(FIELDS))), unpack_moments(value))
        pooled = empty_moments(len(FIELDS))
        for value in stock_moments.values():
            pooled = merge_moments(pooled, value)
        statistics = {"format": "jycapital-normalization-v2", "frequency": frequency, "fields": FIELDS,
                      "fit_interval": interval, "ddof": 0, "rows": fit_rows,
                      "global": pack_moments(pooled),
                      "stocks": {stock: pack_moments(value) for stock, value in sorted(stock_moments.items())}}
        statistics_path = run / "normalization" / f"{frequency}.json"
        atomic_json(statistics_path, statistics)
        statistics_sha = sha256(statistics_path)
        log_event(run, "statistics_frozen", frequency=frequency, rows=fit_rows, stocks=len(stock_moments), sha256=statistics_sha)
        normalizer = Normalizer(statistics, config, training=True)
        records, totals = [], {}
        for source in files:
            destination = run / "data" / frequency / f"{source.stem}.parquet"
            record_path = run / "records" / frequency / f"{source.stem}.json"
            started = time.perf_counter()
            if record_path.exists():
                record = read_json(record_path)
                if record["statistics_sha256"] != statistics_sha or sha256(destination) != record["sha256"]:
                    raise ValueError("已处理分块的统计量或内容发生变化")
                normalizer = Normalizer(statistics, config, training=True, history_dict=record["history"])
            else:
                record = transform_file(connection, source, universe_source, interval, config, normalizer, destination)
                record.update(month=source.stem, statistics_sha256=statistics_sha, history=normalizer.history_dict())
                atomic_json(record_path, record)
                log_event(run, "transformed_month", frequency=frequency, month=source.stem,
                          rows=record["rows"], bytes=record["bytes"], seconds=round(time.perf_counter() - started, 2))
            records.append({key: value for key, value in record.items() if key not in {"history", "columns"}})
            for name in ["rows", "bytes", "missing_input_values", "missing_output_values", "fixed_values", "dynamic_values", "pooled_values", "unavailable_values", "eligible_rows", "cross_sections", "degenerate_cross_section_fields"]:
                totals[name] = totals.get(name, 0) + record.get(name, 0)
        if totals["rows"] != fit_rows:
            raise ValueError("训练拟合与转换的原始行集合不一致")
        summary["frequencies"][frequency] = {"statistics_sha256": statistics_sha, "stocks": len(stock_moments), **totals, "partitions": records}
        atomic_json(run / "summary.json", summary)
        log_event(run, "frequency_completed", frequency=frequency, **totals)
        del normalizer, stock_moments, statistics
        gc.collect()
    manifest.update(status="complete", completed_at=now(), summary_sha256=sha256(run / "summary.json"),
                    data_manifest_sha256=sha256(data_manifest_path))
    atomic_json(run / "manifest.json", manifest)
    log_event(run, "preprocessing_completed", summary=str(run / "summary.json"))
    connection.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=os.environ.get("JYCAPITAL_DATA_DIR", str(ROOT / "data")))
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--protocol", default=str(ROOT / "evaluation" / "protocols" / "local-v1.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare(args.run_dir, args.data_root, read_json(args.config), read_json(args.protocol), args.resume)


if __name__ == "__main__":
    main()
