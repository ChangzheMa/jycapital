"""同日严格后续标签监督；未来真值只出现在损失函数，预测头不接收标签。"""
import torch
from torch import nn
from torch.nn import functional as F


def future_pairs(device=None):
    return torch.triu_indices(8, 8, offset=1, device=device)


class FutureReturnHead(nn.Module):
    def __init__(self, input_width, hidden_width, dropout):
        super().__init__()
        self.projection = nn.Linear(input_width, hidden_width)
        self.target_clock = nn.Embedding(7, hidden_width)
        self.lead = nn.Embedding(7, hidden_width)
        nn.init.normal_(self.target_clock.weight, std=.02)
        nn.init.normal_(self.lead.weight, std=.02)
        self.readout = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_width, 1))
        self.register_buffer('pairs', future_pairs(), persistent=False)

    def forward(self, states):
        if states.ndim != 3 or states.shape[1] != 8:
            raise ValueError('后续收益头要求每个股票日八个独立因果状态')
        query, target = self.pairs
        features = self.projection(states)[:, query]
        features = features + self.target_clock(target - 1) + self.lead(target - query - 1)
        return self.readout(features).squeeze(-1).float()


def future_objective(prediction, targets, sample_weights, specification):
    """先按查询平均有效后续标签损失，再对查询按原日期权重平均。"""
    if targets.ndim != 3 or targets.shape[1:] != (8, 3) or prediction.shape != (targets.shape[0], 28):
        raise ValueError('后续监督要求28个预测与[batch,8,主标签/排名/成员]目标')
    membership = targets[..., 2]
    if not torch.isfinite(membership).all() or not ((membership == 0) | (membership == 1)).all():
        raise ValueError('查询成员必须为显式0/1标志')
    if (torch.isfinite(targets[..., 0]) & ~membership.bool()).any():
        raise ValueError('非成员位置不能携带主标签')
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('后续收益预测非有限')
    if sample_weights is None:
        sample_weights = torch.ones(targets.shape[0], device=targets.device, dtype=torch.float64)
    if sample_weights.shape != (targets.shape[0],) or not torch.isfinite(sample_weights).all() or (sample_weights <= 0).any():
        raise ValueError('日期权重必须为每个股票日一个有限正数')
    query, target = future_pairs(prediction.device)
    future = targets[:, target, 0].double()
    valid = membership[:, query].bool() & torch.isfinite(future)
    safe = torch.where(valid, future, torch.zeros_like(future))
    losses = F.huber_loss(prediction.double(), safe, reduction='none', delta=specification['huber_delta'])
    zeros = F.huber_loss(torch.zeros_like(safe), safe, reduction='none', delta=specification['huber_delta'])
    losses, zeros = losses * valid, zeros * valid
    query_map = F.one_hot(query, num_classes=8).double()
    lead_map = F.one_hot(target-query-1, num_classes=7).double()
    counts = valid.double() @ query_map
    by_query = (losses @ query_map) / counts.clamp_min(1)
    zero_by_query = (zeros @ query_map) / counts.clamp_min(1)
    active = counts > 0
    weights = sample_weights.double()[:, None] * active
    if active.any():
        loss = (by_query * weights).sum() / weights.sum()
    else:
        loss = prediction.double().sum() * 0.
    return loss, {
        'future_queries': int(active.sum()), 'future_pairs': int(valid.sum()),
        'future_query_huber_sum': by_query.detach().sum(),
        'future_zero_query_huber_sum': zero_by_query.detach().sum(),
        'future_lead_huber_sum': (losses.detach().sum(dim=0) @ lead_map),
        'future_lead_zero_huber_sum': (zeros.detach().sum(dim=0) @ lead_map),
        'future_lead_pair_counts': (valid.sum(dim=0).double() @ lead_map).long()}
