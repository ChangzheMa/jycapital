"""单时点内部的时间加权收益、排名、专家及路由目标。"""
import torch
from torch.nn import functional as F
from v3201_objective import balanced_assignment


def clock_objective(prediction, experts, logits, targets, weights, spec):
    if targets.shape != prediction.shape or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError('单时点要求[batch,2]目标')
    if not all(torch.isfinite(x).all() for x in (prediction, experts, logits, targets, weights)):
        raise FloatingPointError('单时点目标或预测非有限')
    if weights.shape != (len(targets),) or not len(targets) or (weights <= 0).any():
        raise ValueError('时间权重必须为非空有限正数')
    weights = weights.double() / weights.double().sum()
    targets, prediction, experts = targets.double(), prediction.double(), experts.double()
    main = F.huber_loss(prediction[:, 0], targets[:, 0], reduction='none', delta=spec['huber_delta'])
    rank = F.huber_loss(prediction[:, 1], targets[:, 1], reduction='none', delta=spec['auxiliary_rank_huber_delta'])
    cost = F.huber_loss(experts[..., 0], targets[:, 0, None].expand_as(experts[..., 0]), reduction='none', delta=spec['huber_delta'])
    cost += spec['auxiliary_rank_weight'] * F.huber_loss(experts[..., 1], targets[:, 1, None].expand_as(experts[..., 1]), reduction='none', delta=spec['auxiliary_rank_huber_delta'])
    assignment = balanced_assignment(cost, spec['assignment_temperature'], spec['assignment_iterations'])
    log_probability = F.log_softmax(logits.double(), dim=-1)
    main_loss, rank_loss = (weights * main).sum(), (weights * rank).sum()
    expert_loss = (weights * (assignment * cost).sum(-1)).sum()
    router_loss = -(weights * (assignment * log_probability).sum(-1)).sum()
    loss = main_loss + spec['auxiliary_rank_weight'] * rank_loss + spec['expert_loss_weight'] * expert_loss + spec['router_loss_weight'] * router_loss
    return loss, {'main_loss': float(main_loss.detach()), 'rank_loss': float(rank_loss.detach()),
                  'expert_loss': float(expert_loss.detach()), 'router_loss': float(router_loss.detach())}
