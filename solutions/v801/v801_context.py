"""冻结本地 GRU 表示，加完整截面的集合交互残差；不使用外部权重。"""

import math

import torch
from torch import nn

from v801_backbone import FrozenBackbone, validate_model_config as validate_backbone
from v801_encoding import encode_all_clocks
from v801_schema import FREQUENCIES, prepare_model_inputs


def validate_model_config(config):
    validate_backbone(config)
    spec = config["model"]["context"]
    for name in ("hidden_dim", "attention_heads", "layers", "latent_tokens", "ff_multiplier"):
        if isinstance(spec[name], bool) or not isinstance(spec[name], int) or spec[name] < 1:
            raise ValueError("集合交互维度和层数必须为正整数")
    if spec["hidden_dim"] % spec["attention_heads"]:
        raise ValueError("集合交互维度须整除注意力头数")
    if spec["complete_cross_section"] is not True or spec["output_initialization"] != "zero_residual":
        raise ValueError("v801 要求完整截面和零残差初始化")
    if not 0 <= spec["dropout"] < 1 or not 0 < spec["residual_scale_initial"] <= 1:
        raise ValueError("非法 dropout 或残差尺度")


class CrossAttention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.heads, self.width = spec["attention_heads"], hidden // spec["attention_heads"]
        self.query, self.key, self.value, self.output = [nn.Linear(hidden, hidden) for _ in range(4)]
        self.dropout = nn.Dropout(spec["dropout"])

    def forward(self, queries, source):
        def split(values):
            return values.reshape(len(values), self.heads, self.width).transpose(0, 1)

        q, k, v = split(self.query(queries)), split(self.key(source)), split(self.value(source))
        # 同一完整截面内汇总；无股票位置编码、股票身份或跨时点混合。
        with torch.autocast(device_type=q.device.type, enabled=False):
            weights = torch.softmax(q.float() @ k.float().transpose(-1, -2) / math.sqrt(self.width), dim=-1)
        mixed = (self.dropout(weights).to(v.dtype) @ v).transpose(0, 1).reshape(len(queries), -1)
        return self.output(mixed)


class SetInteraction(nn.Module):
    def __init__(self, spec):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.tokens = nn.Parameter(torch.randn(spec["latent_tokens"], hidden) * .02)
        self.gather, self.broadcast = CrossAttention(spec), CrossAttention(spec)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * spec["ff_multiplier"]), nn.GELU(),
                                nn.Dropout(spec["dropout"]), nn.Linear(hidden * spec["ff_multiplier"], hidden))
        self.dropout = nn.Dropout(spec["dropout"])
        self.attention_scale = nn.Parameter(torch.tensor(float(spec["residual_scale_initial"])))
        self.ff_scale = nn.Parameter(torch.tensor(float(spec["residual_scale_initial"])))

    def forward(self, stocks):
        market = self.tokens + self.gather(self.tokens, stocks)
        stocks = stocks + self.attention_scale * self.dropout(self.broadcast(stocks, market))
        return stocks + self.ff_scale * self.dropout(self.ff(stocks))


class MultiFrequencyGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_model_config(config)
        self.backbone = FrozenBackbone(config).requires_grad_(False).eval()
        self.feature_dim = len(FREQUENCIES) * config["model"]["hidden_dim"] + 8
        spec = config["model"]["context"]
        hidden = spec["hidden_dim"]
        self.context = nn.Sequential(nn.Linear(self.feature_dim, hidden), nn.GELU(),
                                     *[SetInteraction(spec) for _ in range(spec["layers"])],
                                     nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(spec["dropout"]),
                                     nn.Linear(hidden // 2, 1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_shared(self, windows, latest_clock):
        with torch.no_grad():
            return encode_all_clocks(self.backbone, windows, latest_clock)

    def encode_reference(self, windows, clock):
        with torch.no_grad(), torch.autocast(device_type=clock.device.type, enabled=False):
            prepared, lengths = prepare_model_inputs(windows, clock, self.backbone.window_bars)
            vectors = [self.backbone.encoders[f](prepared[f].float(), lengths[f]) for f in FREQUENCIES]
            return torch.cat([*vectors, clock.float()], dim=1)

    def forward_features(self, features):
        """必须传入同一日期时刻的全部股票，不能按子批重建上下文。"""
        if features.ndim != 2 or features.shape[1] != self.feature_dim or not len(features):
            raise ValueError("集合预测要求非空二维冻结表示")
        clocks = features[:, -8:]
        if not torch.equal(clocks, clocks[:1].expand_as(clocks)):
            raise ValueError("集合交互不能混合多个预测时点")
        with torch.autocast(device_type=features.device.type, enabled=False):
            base = self.backbone.head(features.float()).squeeze(-1)
        return base + self.context(features).squeeze(-1).float()

    def forward(self, windows, clock):
        if isinstance(windows, torch.Tensor):
            return self.forward_features(windows)
        return self.forward_features(self.encode_reference(windows, clock))
