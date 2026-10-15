"""完整截面上的纯 Pearson IC；FP64 统计，退化情况显式返回。"""

from dataclasses import dataclass
import torch


class ConstantPredictionError(ValueError):
    pass


@dataclass
class SectionResult:
    loss: torch.Tensor | None
    ic: torch.Tensor | None
    rows: int
    reason: str | None = None


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
    return (torch.stack([r.loss for r in accepted]).mean(),
            torch.stack([r.ic for r in accepted]).mean())
