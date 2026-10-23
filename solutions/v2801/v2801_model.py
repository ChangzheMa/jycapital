"""冻结因果历史网络与数值树结构的统一fast推理模型。"""
import torch
from torch import nn

from v2801_features import FrozenHistoryEncoder
from v2801_trees import TreeEnsemble


class MultiFrequencyNeuralTree(nn.Module):
    def __init__(self, config, parent_config=None):
        super().__init__()
        parent_config = config['backbone_config'] if parent_config is None else parent_config
        self.encoder = FrozenHistoryEncoder(parent_config)
        if self.encoder.feature_dim != config['model']['feature_dim']:
            raise ValueError('历史编码维度与树特征维度不一致')
        self.trees = TreeEnsemble(self.encoder.feature_dim,config['training']['boosting_rounds'],
                                  config['training']['booster']['max_depth'])
        self.eval()

    def train(self, mode=True):
        if mode:
            raise ValueError('该模型的网络保持固定；树训练使用独立梯度提升入口')
        return super().train(False)

    def forward_group(self, windows, latest_clock):
        features, baseline = self.encoder.group_features(windows,latest_clock)
        correction = self.trees(features.reshape(-1,features.shape[-1])).reshape_as(baseline)
        return (baseline+correction)[...,None]

    def forward_heads(self, windows, clock):
        features, baseline = self.encoder.single_features(windows,clock)
        return (baseline+self.trees(features))[:,None]

    def forward(self, windows, clock):
        return self.forward_heads(windows,clock)[:,0]
