"""八个独立时点读出；预测时刻只用于选择参数，不参与未来标签构造。"""

import copy

import torch
from torch import nn

CLOCKS = ('09:30', '10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30')


class ClockReadout(nn.Module):
    def __init__(self, experts, router):
        super().__init__()
        self.experts = copy.deepcopy(experts)
        self.router = copy.deepcopy(router)

    def forward(self, features):
        experts = torch.stack([expert(features) for expert in self.experts], dim=1).float()
        logits = self.router(features).float()
        probabilities = torch.softmax(logits, dim=-1)
        return (experts * probabilities[..., None]).sum(dim=1), experts, logits


class ClockReadoutBank(nn.Module):
    """共享冻结的384维表示，八套参数互不共享，标签不参与选择读出。"""

    def __init__(self, experts, router):
        super().__init__()
        self.heads = nn.ModuleList([ClockReadout(experts, router) for _ in CLOCKS])

    def forward_group(self, features):
        if features.ndim != 3 or features.shape[1] != len(CLOCKS):
            raise ValueError('要求[股票日,8,表示维度]')
        components = [head(features[:, clock]) for clock, head in enumerate(self.heads)]
        return tuple(torch.stack([value[i] for value in components], dim=1) for i in range(3))

    def forward_clock(self, features, clock_id):
        if type(clock_id) is not int or not 0 <= clock_id < len(CLOCKS):
            raise ValueError('未知预测时点')
        return self.heads[clock_id](features)
