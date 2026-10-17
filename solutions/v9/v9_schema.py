"""v9 固定输入布局：72 个数值、两个组完整标志及一个 Bar 可用标志。"""

VALUE_CHANNELS = 72
NON_BOOK_FIELDS = ("open", "high", "low", "close", "volume", "amount")
BOOK_FIELDS = tuple(
    f"{prefix}{level}"
    for prefix in ("ask_price", "bid_price", "ask_volume", "bid_volume",
                   "ask_num_orders", "bid_num_orders")
    for level in range(1, 6)
)
NON_BOOK_VALUE_INDICES = tuple(range(6)) + tuple(range(36, 42))
BOOK_VALUE_INDICES = tuple(range(6, 36)) + tuple(range(42, 72))
GROUP_MASK_SLICE = slice(72, 74)
BAR_HAS_VALUE_INDEX = 74
INPUT_CHANNELS = 75
FREQUENCIES = ("bar1m", "bar5m", "bar30m")
TRADING_DAY_WINDOWS = {
    "bar1m": {"previous_days": 1, "include_today": True},
    "bar5m": {"previous_days": 5, "include_today": False},
    "bar30m": {"previous_days": 20, "include_today": False},
}
PREDICTION_TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")


def validate_input_config(config):
    data = config.get("data", {})
    expected = {"window_policy": "fixed_trading_days", "trading_day_windows": TRADING_DAY_WINDOWS,
                "window_bars": {"bar1m": 480, "bar5m": 240, "bar30m": 160},
                "calendar_source": "universe_trading_dates", "future_slots": "zero_all_channels",
                "insufficient_history": "left_zero_pad", "time_encoding": "prediction_one_hot_only"}
    if any(data.get(key) != value for key, value in expected.items()) or "history_days" in data:
        raise ValueError("v9 要求固定交易日窗口、未来全零及独立预测时钟，不使用自然日截断或 Bar 时间通道")
    clock = config.get("data", {}).get("prediction_clock", {})
    if clock != {"mode": "one_hot", "times": list(PREDICTION_TIMES), "unknown": "error"} or \
            config.get("model", {}).get("prediction_clock_dim") != len(PREDICTION_TIMES):
        raise ValueError("v9 必须采用固定八时刻 one-hot，未知时刻报错")
    if config.get("solution_id") != "v9" or config.get("model", {}).get("input_channels") != INPUT_CHANNELS:
        raise ValueError("v9 必须使用两个组完整标志及 Bar 可用标志的 75 通道配置")
    spec = config.get("data", {}).get("masking", {})
    if spec.get("mode") != "two_group_complete":
        raise ValueError("v9 掩码语义必须为整组完整才置 1")
    expected = {"non_book": list(NON_BOOK_FIELDS), "book": list(BOOK_FIELDS)}
    if spec.get("groups") != expected:
        raise ValueError("v9 的非盘口与盘口字段分组或顺序已改变")
    if spec.get("value_policy") != "keep_finite_fill_missing_zero" or spec.get("drop_incomplete_samples") is not False:
        raise ValueError("v9 必须保留有效数值和不完整样本")
    if spec.get("bar_validity") != "any_finite_value_in_separate_channel":
        raise ValueError("v9 整根 Bar 的有效性由独立通道表达，且必须与逐数值 any 一致")
