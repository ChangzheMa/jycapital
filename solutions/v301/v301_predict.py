"""从平台逻辑行情源推理；复用冻结统计量与因果切窗，不读取标签。"""

from datetime import date, datetime
import json
from pathlib import Path
import re

import torch
import numpy as np
import pandas as pd

from v301_data import WindowBlock, TradingCalendar, regular_times, minute_slots, VALUE_COLUMNS
from v301_schema import TRADING_DAY_WINDOWS
from v301_model import MultiFrequencyGRU
from v301_preprocessing import (FIELDS, FREQUENCIES, KEYS, Normalizer, UniverseIndex,
                              population_std, empty_moments, log_values, running_moments)


def local_time(value):
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("日期不能为空")
    return value.tz_convert("Asia/Shanghai").tz_localize(None) if value.tzinfo else value


def platform_interval(start_date, end_date):
    start, end = local_time(start_date), local_time(end_date)
    date_only = (isinstance(end_date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_date.strip()) is not None
                 or isinstance(end_date, date) and not isinstance(end_date, datetime))
    stop = end.normalize() + pd.Timedelta(days=1) if date_only else end.floor("min") + pd.Timedelta(minutes=1)
    if start >= stop or (not date_only and start > end):
        raise ValueError("非法推理日期区间")
    return start.ceil("min"), stop


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("非法行情表标识符")
    return value


def read_table(dai_module, table, columns, start, stop, extra=""):
    table = identifier(table)
    columns = [identifier(c) for c in columns]
    query = (f"SELECT {', '.join(columns)} FROM {table} WHERE date >= '{start.isoformat(sep=' ')}' "
             f"AND date < '{stop.isoformat(sep=' ')}' {extra}")
    frame = dai_module.query(query, filters={"date": [start.normalize().strftime("%Y-%m-%d"), stop.ceil("D").strftime("%Y-%m-%d")]}).df()
    frame = frame[columns].copy()
    frame["date"] = pd.to_datetime(frame.date)
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("行情键重复或为空")
    if not frame.empty and not ((frame.date >= start) & (frame.date < stop)).all():
        raise ValueError("行情查询超出区间")
    return frame.sort_values(KEYS).reset_index(drop=True)


def warm_history(dai_module, source, normalizer, start, stop):
    """只为缺少可信自身统计的股票恢复全部已知历史；不计算或依赖未来截面。"""
    if start >= stop:
        return
    complete = {stock for stock, value in normalizer.stocks.items()
                if ((value["count"] >= normalizer.minimum) & (population_std(value) > normalizer.epsilon)).all()}
    observed = dai_module.query(
        f"SELECT DISTINCT instrument FROM {identifier(source)} WHERE date >= '{start.isoformat(sep=' ')}' AND date < '{stop.isoformat(sep=' ')}'",
        filters={"date": [start.normalize().strftime("%Y-%m-%d"), stop.ceil("D").strftime("%Y-%m-%d")]},
    ).df()
    missing = sorted(set(observed.instrument.astype(str)) - complete)
    if not missing:
        return
    quoted = ["'" + stock.replace("'", "''") + "'" for stock in missing]
    extra = "AND instrument IN (" + ",".join(quoted) + ")"
    current = start
    while current < stop:
        end = min((current.to_period("M") + 1).start_time, stop)
        frame = read_table(dai_module, source, KEYS + FIELDS, current, end, extra)
        for stock, part in frame.groupby("instrument", sort=False, observed=True):
            previous = normalizer.history.get(stock)
            if previous is None:
                previous = empty_moments(len(FIELDS)) if normalizer.training else normalizer.stocks.get(stock, empty_moments(len(FIELDS)))
            _, _, _, final = running_moments(log_values(part), previous)
            normalizer.history[stock] = final
        if not frame.empty:
            normalizer.last_time = frame.date.iloc[-1]
        current = end


def prior_trading_days(dai_module, universe_table, start, count):
    """从股票池查询真实交易日期，不用固定自然日或工作日近似。"""
    query = (f"SELECT DISTINCT CAST(date AS DATE) AS date FROM {identifier(universe_table)} "
             f"WHERE date < '{start.normalize().isoformat(sep=' ')}' "
             f"ORDER BY date DESC LIMIT {int(count)}")
    frame = dai_module.query(query).df()
    days = pd.DatetimeIndex(pd.to_datetime(frame.date)).normalize().sort_values().unique()
    if days.hasnans or len(days) > count or (days >= start.normalize()).any():
        raise ValueError("历史交易日查询返回非法日期")
    return days


def predict(datasources, start_date, end_date, model_dir=None, dai_module=None, device=None, bundle=None):
    if dai_module is None:
        import dai as dai_module
    directory = Path(model_dir) if model_dir else Path(__file__).resolve().parent
    if bundle is not None or (directory / "v301-model.json").is_file():
        from v301_text_artifact import decode_state, read_bundle
        bundle = bundle if bundle is not None else read_bundle(directory / "v301-model.json")
        checkpoint = {"config": bundle["config"], "state_dict": decode_state(bundle)}
        preprocessing = bundle["preprocessing"]
    else:
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        preprocessing_path = directory / "preprocessing-config.json"
        if not preprocessing_path.is_file():
            preprocessing_path = directory / "source/solutions/v301/config.json"
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    config = checkpoint["config"]
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MultiFrequencyGRU(config).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    start, stop = platform_interval(start_date, end_date)
    sources = {f: identifier(datasources[f]) for f in FREQUENCIES}
    universe_table = "cpt_jyc_2026_instruments"
    previous = prior_trading_days(dai_module, universe_table, start,
                                  max(s["previous_days"] for s in TRADING_DAY_WINDOWS.values()))
    context_start = previous[0] if len(previous) else start.normalize()
    # 只查更早的最后一个股票池快照，确保首个 Bar 的截面不引用未来池。
    prior = dai_module.query(f"SELECT max(date) AS date FROM {universe_table} WHERE date < '{context_start.isoformat(sep=' ')}'").df().date.iloc[0]
    universe_start = min(context_start, pd.Timestamp(prior)) if pd.notna(prior) else context_start
    universe = read_table(dai_module, universe_table, KEYS, universe_start, stop)
    pool_index = UniverseIndex(universe)
    target_keys = universe.loc[(universe.date >= start) & (universe.date < stop), KEYS]
    if target_keys.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "instrument": pd.Series(dtype="object"),
                             "factor": pd.Series(dtype="float64")})
    context_days = np.sort(universe.loc[universe.date >= context_start, "date"].dt.normalize().unique())
    calendar = TradingCalendar(context_days)
    frequency_starts = {f: calendar.history_start(start, TRADING_DAY_WINDOWS[f]["previous_days"])
                        for f in FREQUENCIES}
    normalizers = {}
    all_stocks = set(universe.instrument.astype(str))
    for frequency in FREQUENCIES:
        if bundle is not None:
            statistics = bundle["normalization"][frequency]
        else:
            statistics_path = directory / "normalization" / (frequency + ".json")
            if not statistics_path.is_file():
                statistics_path = directory / ("normalization-" + frequency + ".json")
            statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        fit_start, fit_stop = (pd.Timestamp(statistics["fit_interval"][k]) for k in ("start", "stop"))
        if start < fit_start:
            raise ValueError("请求早于训练数据起点")
        frequency_start = frequency_starts[frequency]
        normalizer = Normalizer(statistics, preprocessing, training=frequency_start < fit_stop)
        warm_start = fit_start if normalizer.training else fit_stop
        warm_history(dai_module, sources[frequency], normalizer, warm_start, frequency_start)
        normalizers[frequency] = normalizer
        all_stocks.update(normalizer.stocks)
    stocks = pd.Index(sorted(all_stocks))
    trading_days = set(universe.date.dt.normalize())
    # 动态历史保留源表全部日期／时段；模型网格只由股票池交易日构建。
    days = pd.date_range(context_start, stop - pd.Timedelta(1, "ns"), freq="D")
    tails = {f: (np.array([], np.int64), np.empty((0, len(stocks), 72), np.float32)) for f in FREQUENCIES}
    chunks = []
    with torch.inference_mode():
        for day in days:
            lo = pd.Timestamp(day)
            hi = min(lo + pd.Timedelta(days=1), stop)
            keys = target_keys.loc[(target_keys.date >= lo) & (target_keys.date < hi)].reset_index(drop=True)
            grids = {}
            for frequency in FREQUENCIES:
                if lo < frequency_starts[frequency]:
                    grids[frequency] = tails[frequency]
                    continue
                normalizer = normalizers[frequency]
                if lo >= normalizer.fit_stop:
                    normalizer.training = False
                raw = read_table(dai_module, sources[frequency], KEYS + FIELDS, lo, hi)
                normalized, _ = normalizer.transform(raw, pool_index.eligible(raw))
                times = regular_times([day], frequency) if lo in trading_days else np.array([], np.int64)
                # 即便请求截止盘中，仍保留今天完整槽位；未来数值由切窗显式置零。
                values = np.full((len(times), len(stocks), 72), np.nan, np.float32)
                if not normalized.empty:
                    ti = pd.Index(times).get_indexer(normalized.date.to_numpy(dtype="datetime64[ns]").astype(np.int64))
                    si = stocks.get_indexer(normalized.instrument)
                    valid = (ti >= 0) & (si >= 0)
                    values[ti[valid], si[valid]] = normalized.loc[valid, VALUE_COLUMNS].to_numpy(dtype=np.float32)
                old_times, old_values = tails[frequency]
                grids[frequency] = np.concatenate([old_times, times]), np.concatenate([old_values, values])
                length = TRADING_DAY_WINDOWS[frequency]["previous_days"] * len(minute_slots(frequency))
                tails[frequency] = (grids[frequency][0][-length:].copy(), grids[frequency][1][-length:].copy())
                del raw, normalized
            if len(keys):
                block = WindowBlock(keys, grids, stocks, config, context_days)
                factors = np.empty(len(keys), np.float64)
                batch = config["inference"]["batch_size"]
                for offset in range(0, len(keys), batch):
                    selected = np.arange(offset, min(offset + batch, len(keys)))
                    windows, clock = block.batch(selected)
                    windows = {f: torch.from_numpy(v).to(device) for f, v in windows.items()}
                    output = model(windows, torch.from_numpy(clock).to(device))
                    factors[selected] = (output / config["inference"]["factor_scale"]).cpu().numpy()
                keys["factor"] = factors
                chunks.append(keys)
                del block
    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=KEYS + ["factor"])
    result["date"] = pd.to_datetime(result.date)
    result["factor"] = result.factor.astype(np.float64)
    if result.duplicated(KEYS).any() or result[KEYS].isna().any().any() or not np.isfinite(result.factor).all():
        raise ValueError("推理输出键或数值非法")
    if len(result) != len(target_keys):
        raise ValueError("未覆盖完整股票池")
    return result[KEYS + ["factor"]].sort_values(KEYS).reset_index(drop=True)


def main(datasources, start_date, end_date):
    return predict(datasources, start_date, end_date)
