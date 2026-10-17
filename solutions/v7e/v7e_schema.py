"""v7e 固定输入布局：72 个数值、两个组完整标志、三个时间通道。"""

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


def validate_input_config(config):
    if config.get("solution_id") != "v7e" or config.get("model", {}).get("input_channels") != INPUT_CHANNELS:
        raise ValueError("v7e 必须使用两个组完整标志的 77 通道配置")
    spec = config.get("data", {}).get("masking", {})
    if spec.get("mode") != "two_group_complete":
        raise ValueError("v7e 掩码语义必须为整组完整才置 1")
    expected = {"non_book": list(NON_BOOK_FIELDS), "book": list(BOOK_FIELDS)}
    if spec.get("groups") != expected:
        raise ValueError("v7e 的非盘口与盘口字段分组或顺序已改变")
    if spec.get("value_policy") != "keep_finite_fill_missing_zero" or spec.get("drop_incomplete_samples") is not False:
        raise ValueError("v7e 必须保留有效数值和不完整样本")
    if spec.get("bar_validity") != "any_finite_value_encoded_by_nonzero_bar_clock":
        raise ValueError("v7e 整根 Bar 的有效性必须与逐数值 any 一致")
