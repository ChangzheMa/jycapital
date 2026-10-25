"""冻结共享编码器和八个独立的时点收益模型。"""
import torch
from torch import nn

from v3301_features import FrozenSharedEncoder
from v3301_readout import ClockReadoutBank


class MultiFrequencyClockSpecialists(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['solution_id'] != 'v3301' or config['model']['family'] != 'clock_specialists':
            raise ValueError('分时点模型配置不一致')
        spec = config['backbone_config']['model']
        self.encoder = FrozenSharedEncoder(config['backbone_config'])
        experts = nn.ModuleList([nn.Sequential(nn.Linear(spec['fusion_hidden_dim'], spec['expert_hidden_dim']),
            nn.GELU(), nn.Dropout(spec['dropout']), nn.Linear(spec['expert_hidden_dim'], 2))
            for _ in range(spec['expert_count'])])
        router = nn.Sequential(nn.Linear(spec['fusion_hidden_dim'], spec['router_hidden_dim']), nn.GELU(),
                               nn.Linear(spec['router_hidden_dim'], spec['expert_count']))
        self.readouts = ClockReadoutBank(experts, router)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def initialize_from_parent(self, state):
        self.encoder.load_state_dict({k: v for k, v in state.items() if not k.startswith(('experts.', 'router.'))})
        readout = {k: v for k, v in state.items() if k.startswith(('experts.', 'router.'))}
        for head in self.readouts.heads:
            head.load_state_dict(readout)

    def forward_group(self, windows, latest_clock):
        features = self.encoder.group_features(windows, latest_clock)
        return self.readouts.forward_group(features)[0]

    def forward_heads(self, windows, clock):
        features = self.encoder.single_features(windows, clock)
        ids = clock.argmax(dim=1)
        result = features.new_empty((len(features), 2))
        for q in range(8):
            rows = ids == q
            if rows.any():
                result[rows] = self.readouts.forward_clock(features[rows], q)[0]
        return result

    def forward(self, windows, clock):
        return self.forward_heads(windows, clock)[:, 0]
