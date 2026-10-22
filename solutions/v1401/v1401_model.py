"""两个已有冻结 GRU 的组件分数；截面组合在全部股票计算后执行。"""

import torch
from torch import nn

from v1002_model import MultiFrequencyGRU as TemporalModel
from v1101_model import MultiFrequencyGRU as StructuredModel


class FrozenEnsemble(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if [config['solution_id'] for config in configs] != ['v1002', 'v1101']:
            raise ValueError('冻结组合必须按 v1002、v1101 顺序加载')
        if configs[0]['data'] != configs[1]['data']:
            raise ValueError('共享输入要求完全相同的数据契约')
        self.components = nn.ModuleList([TemporalModel(configs[0]), StructuredModel(configs[1])])
        self.requires_grad_(False)

    def component_scores(self, windows, clock):
        return torch.stack([model(windows, clock) for model in self.components], dim=-1)

    def component_scores_group(self, windows, clock):
        return torch.stack([model.forward_group(windows, clock)[..., 0] for model in self.components], dim=-1)
