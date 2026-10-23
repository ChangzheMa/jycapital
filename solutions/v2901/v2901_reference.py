"""保留相对输入，学习可信绝对数量与价格历史的交互。"""
import torch
from torch import nn


def quantity_features(windows):
    if windows.ndim!=3 or windows.shape[1]!=121:
        raise ValueError('数量单位交互要求121通道')
    return windows.transpose(1,2)


class QuantityProjection(nn.Module):
    def __init__(self,output_width,specification,dropout):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(121,specification['width']),nn.GELU(),nn.Dropout(dropout),
                                    nn.Linear(specification['width'],output_width))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.scale=nn.Parameter(torch.tensor(float(specification['residual_scale_initial'])))

    def forward(self,features):
        if features.ndim!=3 or features.shape[-1]!=121:
            raise ValueError('数量单位交互输入形状不一致')
        available=features[...,99:121].gt(0).any(dim=-1,keepdim=True)
        return self.scale*self.network(features)*available
