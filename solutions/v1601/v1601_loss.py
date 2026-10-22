"""原收益主目标和排名辅助目标的 Huber 回归；IC 使用原始收益标签。"""

from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


class ConstantPredictionError(ValueError):
    pass


@dataclass
class SectionResult:
    loss: torch.Tensor | None
    ic: torch.Tensor | None
    rows: int
    reason: str | None = None


def standardized_targets(labels, min_count=30):
    """用一个完整截面的有限标签构造目标；无效位置保留 NaN。"""
    if labels.ndim != 1:
        raise ValueError("标准化标签要求一维完整截面")
    labels = labels.detach().to(torch.float64)
    eligible = torch.isfinite(labels)
    count = int(eligible.sum())
    target = torch.full_like(labels, float("nan"))
    if count < min_count:
        return target, "insufficient_labels"
    centered = labels[eligible] - labels[eligible].mean()
    scale = centered.abs().amax()
    if not torch.isfinite(scale):
        raise FloatingPointError("非有限标签中心化尺度")
    if scale == 0:
        return target, "constant_labels"
    # 等价于除以总体标准差；避免极小收益平方下溢，不加 epsilon。
    unit = centered / scale
    target[eligible] = unit / unit.square().mean().sqrt()
    return target, None


def sample_huber_loss(scores, targets, delta=1.0):
    """目标已由完整截面构造；随机样本 batch 内只计算逐样本 Huber。"""
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta 必须为有限正数")
    if scores.ndim != 1 or targets.shape != scores.shape or not scores.numel():
        raise ValueError("样本 Huber 要求等长的非空一维预测和目标")
    if not torch.isfinite(scores).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("非有限预测或标准化目标")
    with torch.autocast(device_type=scores.device.type, enabled=False):
        loss = F.huber_loss(scores.to(torch.float64), targets.to(torch.float64), delta=delta)
    if not torch.isfinite(loss):
        raise FloatingPointError("非有限 Huber 损失")
    return loss


def rank_standardized_targets(labels, min_count=30):
    """有限标签的平均秩按总体标准差归一化；相同收益获得相同目标。"""
    if labels.ndim != 1:
        raise ValueError("排名目标要求一维完整截面")
    labels = labels.detach().to(torch.float64)
    eligible = torch.isfinite(labels)
    target = torch.full_like(labels, float("nan"))
    if int(eligible.sum()) < min_count:
        return target, "insufficient_labels"
    values, order = torch.sort(labels[eligible])
    _, inverse, counts = torch.unique_consecutive(values, return_inverse=True, return_counts=True)
    if counts.numel() == 1:
        return target, "constant_labels"
    ends = counts.cumsum(0).to(torch.float64)
    average_ranks = (ends + ends - counts - 1) / 2
    ranks = torch.empty_like(values)
    ranks[order] = average_ranks[inverse]
    centered = ranks - ranks.mean()
    target[eligible] = centered / centered.square().mean().sqrt()
    return target, None


def rank_standardized_huber_loss(scores, labels, min_count=30, delta=1.0):
    """训练目标使用排名；诊断 Pearson IC 不变换收益标签。"""
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("排名 Huber 要求同一完整截面的等长一维预测和标签")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("非有限预测")
    with torch.autocast(device_type=scores.device.type, enabled=False):
        target, reason = rank_standardized_targets(labels, min_count)
        count = int(torch.isfinite(labels).sum())
        if reason:
            return SectionResult(None, None, count, reason)
        eligible = torch.isfinite(target)
        loss = sample_huber_loss(scores[eligible], target[eligible], delta)
        try:
            ic = pearson_ic_loss(scores.detach(), labels, min_count).ic
        except ConstantPredictionError:
            ic = None
        return SectionResult(loss, ic, count)


def standardized_huber_loss(scores, labels, min_count=30, delta=1.0):
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Huber 损失要求同一完整截面的等长一维分数和标签")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("非有限预测")
    with torch.autocast(device_type=scores.device.type, enabled=False):
        target, reason = standardized_targets(labels, min_count)
        eligible = torch.isfinite(target)
        count = int(torch.isfinite(labels).sum())
        if reason:
            return SectionResult(None, None, count, reason)
        loss = sample_huber_loss(scores[eligible], target[eligible], delta)
        try:
            ic = pearson_ic_loss(scores.detach(), labels, min_count).ic
        except ConstantPredictionError:
            # 常数预测的 Huber 损失和梯度仍有定义，不因 IC 无定义而跳过更新。
            ic = None
        return SectionResult(loss, ic, count)


def pearson_ic_loss(scores, labels, min_count=30):
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("IC 损失要求同一完整截面的等长一维分数和标签")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("非有限预测")
    with torch.autocast(device_type=scores.device.type, enabled=False):
        labels = labels.to(torch.float64)
        eligible = torch.isfinite(labels)
        count = int(eligible.sum())
        if count < min_count:
            return SectionResult(None, None, count, "insufficient_labels")
        u = scores[eligible].to(torch.float64)
        v = labels[eligible]
        u = u - u.mean()
        v = v - v.mean()
        unorm, vnorm = torch.linalg.vector_norm(u), torch.linalg.vector_norm(v)
        if not torch.isfinite(unorm) or not torch.isfinite(vnorm):
            raise FloatingPointError("非有限截面范数")
        if vnorm == 0:
            return SectionResult(None, None, count, "constant_labels")
        if unorm == 0:
            raise ConstantPredictionError("非结构性常数预测，不能丢弃该截面")
        ic = ((u / unorm) * (v / vnorm)).sum()
        if not torch.isfinite(ic):
            raise FloatingPointError("非有限 IC")
        return SectionResult(1.0 - ic, ic, count)


def mean_section_loss(results):
    accepted = [result for result in results if result.loss is not None]
    if not accepted:
        return None, None
    defined = [r.ic for r in accepted if r.ic is not None]
    return (torch.stack([r.loss for r in accepted]).mean(),
            torch.stack(defined).mean() if defined else None)
