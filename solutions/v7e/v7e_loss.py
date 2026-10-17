"""截面 z-score 标签的 Huber 回归；原始预测不归一化，IC 单独报告。"""

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


def standardized_huber_loss(scores, labels, min_count=30, delta=1.0):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta 必须为有限正数")
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Huber 损失要求同一完整截面的等长一维分数和标签")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("非有限预测")
    with torch.autocast(device_type=scores.device.type, enabled=False):
        labels = labels.detach().to(torch.float64)
        eligible = torch.isfinite(labels)
        count = int(eligible.sum())
        if count < min_count:
            return SectionResult(None, None, count, "insufficient_labels")
        centered = labels[eligible] - labels[eligible].mean()
        scale = centered.abs().amax()
        if not torch.isfinite(scale):
            raise FloatingPointError("非有限标签中心化尺度")
        if scale == 0:
            return SectionResult(None, None, count, "constant_labels")
        # 等价于除以总体标准差；先缩放，避免极小收益平方时下溢，不加 epsilon。
        unit = centered / scale
        target = unit / unit.square().mean().sqrt()
        loss = F.huber_loss(scores[eligible].to(torch.float64), target, delta=delta)
        if not torch.isfinite(loss):
            raise FloatingPointError("非有限 Huber 损失")
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
