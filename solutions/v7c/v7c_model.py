"""四频 TCN 使用窗口内双向卷积；整个输入窗口截至预测时刻。"""

import torch
from torch import nn
from torch.nn import functional as F
from v7c_schema import INPUT_CHANNELS, BAR_CLOCK_SLICE, validate_input_config

FREQUENCIES = ("bar1m", "bar5m", "bar15m", "bar30m")


class SymmetricBlock(nn.Module):
    def __init__(self, hidden, kernel, dilation, dropout):
        super().__init__()
        self.padding = (kernel - 1) * dilation // 2
        self.convolutions = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel, dilation=dilation) for _ in range(2)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        residual = x
        for conv in self.convolutions:
            x = conv(F.pad(x, (self.padding, self.padding), mode="replicate"))
            x = self.dropout(F.gelu(x)) * valid
        return (x + residual) * valid


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, blocks):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.project = nn.Conv1d(INPUT_CHANNELS, hidden, 1)
        self.blocks = nn.ModuleList([
            SymmetricBlock(hidden, spec["kernel_size"], 2 ** i, spec["tcn_dropout"])
            for i in range(blocks)
        ])
        self.pool = nn.Linear(2 * hidden, hidden)

    def forward(self, x):
        # 切窗保证：任一字段有效时 sin/cos 不同时为零，全无效时二者都置零。
        # 组完整标志不能用作整根 Bar 的屏蔽条件，否则会抹掉部分有效的数据。
        valid = (x[:, BAR_CLOCK_SLICE].abs().sum(dim=1, keepdim=True) > 0).to(x.dtype)
        h = F.gelu(self.project(x)) * valid
        for block in self.blocks:
            h = block(h, valid)
        mean = h.sum(dim=2) / valid.sum(dim=2).clamp_min(1)
        positions = torch.arange(h.shape[2], device=h.device)[None, :]
        last_index = torch.where(valid[:, 0].bool(), positions, -1).amax(dim=1)
        last = h.gather(2, last_index.clamp_min(0)[:, None, None].expand(-1, h.shape[1], 1)).squeeze(2)
        pooled = F.gelu(self.pool(torch.cat([last, mean], dim=1)))
        return pooled * (last_index >= 0)[:, None]


class MultiFrequencyTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        spec = config["model"]
        validate_input_config(config)
        if spec.get("padding") != "symmetric_replicate" or spec["kernel_size"] % 2 != 1:
            raise ValueError("v7c 要求奇数核宽和左右对称边界复制")
        self.encoders = nn.ModuleDict({
            f: FrequencyEncoder(spec, spec["blocks"][f]) for f in FREQUENCIES
        })
        self.head = nn.Sequential(
            nn.Linear(4 * spec["hidden_dim"] + 2, spec["fusion_hidden_dim"]),
            nn.GELU(), nn.Dropout(spec["dropout"]), nn.Linear(spec["fusion_hidden_dim"], 1),
        )

    def forward(self, windows, clock):
        vectors = [self.encoders[f](windows[f]) for f in FREQUENCIES]
        return self.head(torch.cat([*vectors, clock], dim=1)).squeeze(1)
