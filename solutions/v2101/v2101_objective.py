"""收益主任务与数量辅助监督；主梯度保持，辅助梯度仅移除冲突分量。"""
import torch
from torch.nn import functional as F

from v2101_quantity import GROUPS, unpack_targets

SCALAR_STATISTICS = ("quantity_loss", "gradient_cosine", "gradient_conflict", "auxiliary_scale",
    "primary_gradient_norm", "auxiliary_gradient_norm", "projected_auxiliary_dot_primary",
    "quantity_path_rms", "prediction_rms", "quantity_path_centered_rms", "prediction_centered_rms",
    "return_gate_entropy", "quantity_gate_entropy")
SUM_STATISTICS = ("quantity_model_sse", "quantity_last_sse", "quantity_same_slot_sse", "quantity_zero_sse",
                  "quantity_matched_values", "quantity_valid_values") + tuple(
    "quantity_" + group + "_" + metric for group in ("trades", "book_volume", "book_order_count")
    for metric in ("model_sse", "last_sse", "same_slot_sse", "zero_sse", "matched_values"))


def task_objectives(tasks, targets, sample_weights, spec):
    primary, future, latest, same_slot = unpack_targets(targets)
    prediction, delta = tasks["prediction"].double(), tasks["quantity_delta"].double()
    if prediction.shape != primary.shape or delta.shape != future.shape:
        raise ValueError("主／辅输出形状不符")
    if not torch.isfinite(prediction).all() or not torch.isfinite(delta).all():
        raise FloatingPointError("模型输出含非有限值")
    valid = torch.isfinite(primary[..., 0])
    if not torch.equal(valid, torch.isfinite(primary[..., 1])) or not valid.any():
        raise ValueError("收益和排名目标须共享非空有效位置")
    if sample_weights is None:
        sample_weights = torch.ones(len(targets), dtype=torch.float64, device=targets.device)
    if sample_weights.shape != (len(targets),) or not torch.isfinite(sample_weights).all() or (sample_weights <= 0).any():
        raise ValueError("日期权重须为有限正数")
    weights = sample_weights.double()[:, None].expand(-1, 8)[valid]
    weights = weights / weights.sum()
    main = (weights * F.huber_loss(prediction[..., 0][valid], primary[..., 0][valid],
                                  reduction="none", delta=spec["huber_delta"])).sum()
    rank = (weights * F.huber_loss(prediction[..., 1][valid], primary[..., 1][valid],
                                  reduction="none", delta=spec["auxiliary_rank_huber_delta"])).sum()
    primary_loss = main + spec["auxiliary_rank_weight"] * rank
    quantity_valid = torch.isfinite(future) & torch.isfinite(latest)
    change = torch.where(quantity_valid, future - latest, 0)
    error = F.huber_loss(delta, change, reduction="none", delta=spec["quantity_huber_delta"])
    quantity_weights = sample_weights.double()[:, None, None].expand_as(future)
    groups = []
    for start, stop in GROUPS:
        usable = quantity_valid[..., start:stop]
        group_weights = quantity_weights[..., start:stop] * usable
        if usable.any():
            groups.append((error[..., start:stop] * group_weights).sum() / group_weights.sum())
    auxiliary_loss = torch.stack(groups).mean() if groups else delta.sum() * 0
    if not torch.isfinite(primary_loss) or not torch.isfinite(auxiliary_loss):
        raise FloatingPointError("非有限主／辅助损失")
    matched = quantity_valid & torch.isfinite(same_slot)
    safe_error = torch.where(matched, delta.detach() - change, 0)
    last_error = torch.where(matched, change, 0)
    old_error = torch.where(matched, same_slot - future, 0)
    zero_error = torch.where(matched, future, 0)
    path = prediction.detach()[..., 0] - tasks["without_quantity_prediction"].detach().double()[..., 0]
    details = {"main_loss": main.detach(), "rank_loss": rank.detach(), "quantity_loss": auxiliary_loss.detach(),
               "quantity_model_sse": safe_error.square().sum(), "quantity_last_sse": last_error.square().sum(),
               "quantity_same_slot_sse": old_error.square().sum(), "quantity_matched_values": int(matched.sum()),
               "quantity_zero_sse": zero_error.square().sum(),
               "quantity_valid_values": int(quantity_valid.sum()), "quantity_path_rms": path.square().mean().sqrt(),
               "prediction_rms": prediction.detach()[..., 0].square().mean().sqrt(), "rows": int(valid.sum())}
    details["quantity_path_centered_rms"] = (path - path.mean(dim=0, keepdim=True)).square().mean().sqrt()
    scores = prediction.detach()[..., 0]
    details["prediction_centered_rms"] = (scores - scores.mean(dim=0, keepdim=True)).square().mean().sqrt()
    for name, (start, stop) in zip(("trades", "book_volume", "book_order_count"), GROUPS):
        for baseline, errors in (("model", safe_error), ("last", last_error), ("same_slot", old_error), ("zero", zero_error)):
            details["quantity_" + name + "_" + baseline + "_sse"] = errors[..., start:stop].square().sum()
        details["quantity_" + name + "_matched_values"] = int(matched[..., start:stop].sum())
    for name in ("return", "quantity"):
        probability = tasks[name + "_gate"].detach().double()
        details[name + "_gate_entropy"] = -(probability * probability.clamp_min(1e-30).log()).sum(dim=-1).mean()
    return primary_loss, auxiliary_loss, details


def combine_gradients(parameters, primary, auxiliary, weight, cap):
    """一次确定性投影；非对称主任务优先变体，不声称等同完整PCGrad。"""
    parameters = list(parameters)
    if len(parameters) != len(primary) or len(parameters) != len(auxiliary) or not 0 <= weight <= 1 or cap <= 0:
        raise ValueError("梯度数量或组合参数非法")
    primary = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, primary)]
    auxiliary = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, auxiliary)]
    # dot/范数只用于选择系数，以FP64累加；梯度本体仍为参数的FP32。
    norm_primary = sum((g.double().square().sum() for g in primary))
    norm_auxiliary = sum((g.double().square().sum() for g in auxiliary))
    dot = sum((g.double().mul(h).sum() for g, h in zip(primary, auxiliary)))
    if not all(torch.isfinite(value) for value in (norm_primary, norm_auxiliary, dot)):
        raise FloatingPointError("非有限任务梯度")
    coefficient = torch.minimum(dot, torch.zeros_like(dot)) / norm_primary if norm_primary > 0 else dot * 0
    projected = [h - coefficient.to(g.dtype) * g for g, h in zip(primary, auxiliary)]
    projected_norm = sum((h.double().square().sum() for h in projected))
    scale = torch.minimum(torch.ones_like(dot), cap * (norm_primary / projected_norm).sqrt()) if projected_norm > 0 else dot * 0
    post_dot = sum((g.double().mul(h).sum() for g, h in zip(primary, projected)))
    for parameter, g, h in zip(parameters, primary, projected):
        parameter.grad = g + (weight * scale).to(g.dtype) * h
    cosine = dot / (norm_primary * norm_auxiliary).sqrt() if norm_primary > 0 and norm_auxiliary > 0 else dot * 0
    return {"gradient_cosine": cosine, "gradient_conflict": (dot < 0).double(), "auxiliary_scale": scale,
            "primary_gradient_norm": norm_primary.sqrt(), "auxiliary_gradient_norm": norm_auxiliary.sqrt(),
            "projected_auxiliary_dot_primary": post_dot}
