"""共享于所有股票的四频因果 TCN；每个样本仅包含一只股票。"""

import torch
from torch import nn
from torch.nn import functional as F

FREQUENCIES = ("bar1m", "bar5m", "bar15m", "bar30m")


class CausalBlock(nn.Module):
    def __init__(self, hidden, kernel, dilation, dropout):
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.convolutions = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel, dilation=dilation) for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        residual = x
        for conv, norm in zip(self.convolutions, self.norms):
            x = conv(F.pad(x, (self.left, 0)))
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = self.dropout(F.gelu(x)) * valid
        return (x + residual) * valid


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, blocks):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.project = nn.Conv1d(147, hidden, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(hidden, spec["kernel_size"], 2 ** i, spec["dropout"])
            for i in range(blocks)
        ])
        self.pool = nn.Linear(2 * hidden, hidden)

    def forward(self, x):
        valid = (x[:, 72:144].sum(dim=1, keepdim=True) > 0).to(x.dtype)
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
