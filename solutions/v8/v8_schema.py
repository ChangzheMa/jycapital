"""v8 固定输入布局：72 个数值、两个组完整标志、三个时间通道。"""

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
BAR_CLOCK_SLICE = slice(74, 76)
LOG_AGE_INDEX = 76
INPUT_CHANNELS = 77
PREDICTION_TIMES = ("09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30")


def validate_input_config(config):
    if config.get("data", {}).get("time_encoding") != "bar_clock_sin_cos_log_age_and_prediction_one_hot":
        raise ValueError("v8 保留 Bar 时间通道，仅将预测时刻替换为 one-hot")
    clock = config.get("data", {}).get("prediction_clock", {})
    if clock != {"mode": "one_hot", "times": list(PREDICTION_TIMES), "unknown": "error"} or \
            config.get("model", {}).get("prediction_clock_dim") != len(PREDICTION_TIMES):
        raise ValueError("v8 必须采用固定八时刻 one-hot，未知时刻报错")
    if config.get("solution_id") != "v8" or config.get("model", {}).get("input_channels") != INPUT_CHANNELS:
        raise ValueError("v8 必须使用两个组完整标志的 77 通道配置")
    spec = config.get("data", {}).get("masking", {})
    if spec.get("mode") != "two_group_complete":
        raise ValueError("v8 掩码语义必须为整组完整才置 1")
    expected = {"non_book": list(NON_BOOK_FIELDS), "book": list(BOOK_FIELDS)}
    if spec.get("groups") != expected:
        raise ValueError("v8 的非盘口与盘口字段分组或顺序已改变")
    if spec.get("value_policy") != "keep_finite_fill_missing_zero" or spec.get("drop_incomplete_samples") is not False:
        raise ValueError("v8 必须保留有效数值和不完整样本")
    if spec.get("bar_validity") != "any_finite_value_encoded_by_nonzero_bar_clock":
        raise ValueError("v8 整根 Bar 的有效性必须与逐数值 any 一致")
