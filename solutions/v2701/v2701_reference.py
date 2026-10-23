"""成交过程与同一已闭合Bar日级参考的联合输入投影。"""
import torch
from torch import nn


def reference_features(windows):
    if windows.ndim != 3 or windows.shape[1] != 83:
        raise ValueError('成交参考要求83通道')
    raw = windows[:, :75, :].transpose(1, 2)
    extra = windows[:, 77:83, :].transpose(1, 2)
    valid_pair = (windows[:, 76, :] > 0) & (windows[:, 81, :] > 0)
    distance = 100. * (windows[:, 75, :].float() - windows[:, 78, :].float())
    distance = torch.where(valid_pair, distance, torch.zeros_like(distance))
    # 保留父视图，追加日级共同中心。它是输入坐标，不生成收益监督或复权价格。
    return torch.cat((raw, extra, distance[..., None], valid_pair.float()[..., None]), dim=-1)


class ReferenceProjection(nn.Module):
    def __init__(self, output_width, specification, dropout):
        super().__init__()
        width = specification['width']
        self.network = nn.Sequential(nn.Linear(83, width), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(width, output_width))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(specification['residual_scale_initial'])))

    def forward(self, features):
        if features.ndim != 3 or features.shape[-1] != 83:
            raise ValueError('成交与参考投影输入形状不一致')
        available = features[..., 78:81].gt(0).any(dim=-1, keepdim=True)
        return self.scale * self.network(features) * available
