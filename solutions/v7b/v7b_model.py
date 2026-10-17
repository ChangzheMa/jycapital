"""共享于所有股票的四频因果 TCN；每个样本仅包含一只股票。"""

import torch
from torch import nn
from torch.nn import functional as F
from v7b_schema import INPUT_CHANNELS, validate_input_config

FREQUENCIES = ("bar1m", "bar5m", "bar15m", "bar30m")


class CausalBlock(nn.Module):
    def __init__(self, hidden, kernel, dilation, dropout):
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.convolutions = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel, dilation=dilation) for _ in range(2)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        for conv in self.convolutions:
            x = conv(F.pad(x, (self.left, 0)))
            x = self.dropout(F.gelu(x))
        return x + residual


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, blocks, window_bars):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.project = nn.Conv1d(INPUT_CHANNELS, hidden, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(hidden, spec["kernel_size"], 2 ** i, spec["tcn_dropout"])
            for i in range(blocks)
        ])
        self.window_bars = window_bars
        self.pool = nn.Linear(hidden * window_bars, hidden)

    def forward(self, x):
        if x.shape[2] != self.window_bars:
            raise ValueError("展平输入长度必须与该分支配置的窗口长度一致")
        h = F.gelu(self.project(x))
        for block in self.blocks:
            h = block(h)
        return F.gelu(self.pool(h.flatten(start_dim=1)))


class MultiFrequencyTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        spec = config["model"]
        validate_input_config(config)
        if spec.get("hidden_validity_mask") is not False or spec.get("pooling") != "flatten":
            raise ValueError("v7b 要求同时取消隐藏层掩码并展平分支特征")
        self.encoders = nn.ModuleDict({
            f: FrequencyEncoder(spec, spec["blocks"][f], config["data"]["window_bars"][f])
            for f in FREQUENCIES
        })
        self.head = nn.Sequential(
            nn.Linear(4 * spec["hidden_dim"] + 2, spec["fusion_hidden_dim"]),
            nn.GELU(), nn.Dropout(spec["dropout"]), nn.Linear(spec["fusion_hidden_dim"], 1),
        )

    def forward(self, windows, clock):
        vectors = [self.encoders[f](windows[f]) for f in FREQUENCIES]
        return self.head(torch.cat([*vectors, clock], dim=1)).squeeze(1)
