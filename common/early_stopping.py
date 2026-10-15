"""后续实验统一使用的动态早停：最佳 epoch 的 60%，向上取整并限制为 4—12。"""

import math


EARLY_STOPPING_POLICY = {
    "mode": "best_epoch_fraction",
    "fraction": 0.6,
    "minimum": 4,
    "maximum": 12,
    "rounding": "ceil",
    "improvement": "strict",
}


def validate_early_stopping(training):
    if "patience" in training or "min_delta" in training:
        raise ValueError("动态早停不接受固定 patience 或 min_delta；验证主指标严格创新高即重置计数")
    if training.get("early_stopping") != EARLY_STOPPING_POLICY:
        raise ValueError("早停须使用最佳 epoch × 0.6 向上取整、最少 4 轮、最多 12 轮的统一配置")


def patience_epochs(best_epoch):
    """epoch 从 1 开始；尚无合格最佳结果时 best_epoch=0，耐心仍为 4。"""
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 0:
        raise ValueError("最佳 epoch 必须为非负整数")
    # ceil(best_epoch * 3 / 5)，使用整数运算避免浮点舍入改变边界。
    return min(12, max(4, (best_epoch * 3 + 4) // 5))


def select_epoch(state, metric, epoch, eligible=True):
    """合格的严格新高同时更新最佳权重轮次与早停计数；平分或不合格均计为无提升。"""
    if metric is None or not math.isfinite(metric):
        raise ValueError("验证主指标无有效值")
    improved = eligible and metric > state["best"]
    if improved:
        state.update(best=metric, best_epoch=epoch, stale=0)
    else:
        state["stale"] += 1
    return improved


def should_stop(state):
    return state["stale"] >= patience_epochs(state["best_epoch"])
