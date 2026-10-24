"""仅训练使用的时间加权损失与近似平衡分支分配。"""

import torch
from torch.nn import functional as F


@torch.no_grad()
def balanced_assignment(costs, temperature, iterations):
    if costs.ndim != 2 or min(costs.shape) < 1 or not torch.isfinite(costs).all():
        raise ValueError("分配代价须为有限的样本×分支矩阵")
    values = -costs.detach().double() / temperature
    values -= values.max(dim=1, keepdim=True).values
    log_column_mass = torch.log(torch.tensor(costs.shape[0] / costs.shape[1],
                                            dtype=torch.float64, device=costs.device))
    for _ in range(iterations):
        values -= torch.logsumexp(values, dim=1, keepdim=True)
        values -= torch.logsumexp(values, dim=0, keepdim=True)
        values += log_column_mass
    # 有限次迭代：行归一化精确到浮点精度，列负载仅近似平衡。
    return torch.softmax(values, dim=1)


def combined_objective(prediction, experts, logits, targets, sample_weights, spec):
    if targets.shape != prediction.shape or targets.ndim != 3 or targets.shape[1:] != (8, 2):
        raise ValueError("股票日训练要求 [batch, 8, 2] 双头目标")
    valid = torch.isfinite(targets[..., 0])
    if not torch.equal(valid, torch.isfinite(targets[..., 1])) or not valid.any():
        raise ValueError("双头目标必须共享非空有效样本")
    if sample_weights is None:
        # 仅独立合成预检使用单位权重；正式训练入口显式传入日期权重。
        sample_weights = torch.ones(targets.shape[0], device=targets.device, dtype=torch.float64)
    if sample_weights.shape != (targets.shape[0],) or not torch.isfinite(sample_weights).all() or (sample_weights <= 0).any():
        raise ValueError("时间权重须为每个股票日一个有限正数")
    if not all(torch.isfinite(x).all() for x in (prediction, experts, logits)):
        raise FloatingPointError("非有限模型输出")
    weights = sample_weights.double()[:, None].expand(-1, 8)[valid]
    weights = weights / weights.sum()
    main_target, rank_target = targets[..., 0][valid].double(), targets[..., 1][valid].double()
    main = F.huber_loss(prediction[..., 0][valid].double(), main_target, reduction="none", delta=spec["huber_delta"])
    rank = F.huber_loss(prediction[..., 1][valid].double(), rank_target, reduction="none", delta=spec["auxiliary_rank_huber_delta"])
    selected_experts = experts[valid].double()
    cost = F.huber_loss(selected_experts[..., 0], main_target[:, None].expand_as(selected_experts[..., 0]),
                        reduction="none", delta=spec["huber_delta"])
    cost += spec["auxiliary_rank_weight"] * F.huber_loss(
        selected_experts[..., 1], rank_target[:, None].expand_as(selected_experts[..., 1]),
        reduction="none", delta=spec["auxiliary_rank_huber_delta"])
    assignment = balanced_assignment(cost, spec["assignment_temperature"], spec["assignment_iterations"])
    log_probability = F.log_softmax(logits[valid].double(), dim=-1)
    main_loss, rank_loss = (weights * main).sum(), (weights * rank).sum()
    expert_loss = (weights * (assignment * cost).sum(dim=-1)).sum()
    router_loss = -(weights * (assignment * log_probability).sum(dim=-1)).sum()
    loss = (main_loss + spec["auxiliary_rank_weight"] * rank_loss
            + spec["expert_loss_weight"] * expert_loss + spec["router_loss_weight"] * router_loss)
    probability = log_probability.exp().detach()
    return loss, {"main_loss": main_loss.detach(), "rank_loss": rank_loss.detach(),
                  "expert_loss": expert_loss.detach(), "router_loss": router_loss.detach(),
                  "routing_usage": probability.mean(dim=0),
                  "routing_entropy": -(probability * log_probability.detach()).sum(dim=-1).mean(),
                  "rows": int(valid.sum())}
