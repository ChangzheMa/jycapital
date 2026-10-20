"""先学习时间 patch 投影，再进行低秩时间／通道混合。"""

import torch
from torch import nn
from torch.nn import functional as F

from v202_schema import (FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs,
                         validate_backbone_config)


def validate_model_config(config):
    validate_backbone_config(config, "factorized_mixer", ("layers", "temporal_rank", "channel_hidden_dim", "readout_tokens"))
    spec = config["model"]
    expected = {"normalization": "pre_layernorm", "pooling": "patch_factorized_temporal_projection",
                "hidden_mask": "future_patches_only"}
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("v202 要求 Pre-LN、patch 分解汇总与未来 patch 掩码")
    patches = spec.get("patch_bars", {})
    if set(patches) != set(FREQUENCIES):
        raise ValueError("每个频率必须配置 patch 长度")
    for frequency, size in patches.items():
        if isinstance(size, bool) or not isinstance(size, int) or size < 1 or config["data"]["window_bars"][frequency] % size:
            raise ValueError("patch 长度必须为整除窗口的正整数")
    token_lengths = [config["data"]["window_bars"][f] // patches[f] for f in FREQUENCIES]
    if max(spec["temporal_rank"], spec["readout_tokens"]) > min(token_lengths):
        raise ValueError("低秩时间维和汇总槽位不得超过最短分支窗口")


class MixerBlock(nn.Module):
    def __init__(self, spec, length):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.time_norm, self.channel_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.time = nn.Sequential(nn.Linear(length, spec["temporal_rank"]), nn.GELU(),
                                  nn.Dropout(spec["dropout"]), nn.Linear(spec["temporal_rank"], length))
        self.channel = nn.Sequential(nn.Linear(hidden, spec["channel_hidden_dim"]), nn.GELU(),
                                     nn.Dropout(spec["dropout"]), nn.Linear(spec["channel_hidden_dim"], hidden))
        self.dropout = nn.Dropout(spec["dropout"])

    def forward(self, x, available):
        normalized = self.time_norm(x).masked_fill(~available, 0)
        mixed = self.time(normalized.transpose(1, 2)).transpose(1, 2)
        x = (x + self.dropout(mixed)).masked_fill(~available, 0)
        return (x + self.dropout(self.channel(self.channel_norm(x)))).masked_fill(~available, 0)


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, frequency, length):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.patch = spec["patch_bars"][frequency]
        self.length = length // self.patch
        self.project = nn.Linear(INPUT_CHANNELS * self.patch, hidden)
        self.blocks = nn.ModuleList([MixerBlock(spec, self.length) for _ in range(spec["layers"])])
        self.norm = nn.LayerNorm(hidden)
        # 同一时间投影由全部隐藏通道共享，避免 hidden × length 到 hidden 的大矩阵。
        self.time_pool = nn.Linear(self.length, spec["readout_tokens"])
        self.pool = nn.Linear(hidden * spec["readout_tokens"], hidden)

    def forward(self, x, lengths):
        available = (torch.arange(self.length, device=x.device)[None, :] * self.patch < lengths[:, None])[:, :, None]
        patches = x.transpose(1, 2).reshape(len(x), self.length, INPUT_CHANNELS * self.patch)
        h = F.gelu(self.project(patches)).masked_fill(~available, 0)
        for block in self.blocks:
            h = block(h, available)
        h = self.norm(h).masked_fill(~available, 0)
        return F.gelu(self.pool(self.time_pool(h.transpose(1, 2)).flatten(start_dim=1)))


class MultiFrequencyMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_model_config(config)
        spec = config["model"]
        self.window_bars = dict(config["data"]["window_bars"])
        self.encoders = nn.ModuleDict({f: FrequencyEncoder(spec, f, self.window_bars[f]) for f in FREQUENCIES})
        self.head = nn.Sequential(
            nn.Linear(len(FREQUENCIES) * spec["hidden_dim"] + spec["prediction_clock_dim"], spec["fusion_hidden_dim"]),
            nn.GELU(), nn.Dropout(spec["dropout"]), nn.Linear(spec["fusion_hidden_dim"], 1))

    def forward(self, windows, clock):
        windows, lengths = prepare_model_inputs(windows, clock, self.window_bars)
        vectors = [self.encoders[f](windows[f], lengths[f]) for f in FREQUENCIES]
        return self.head(torch.cat([*vectors, clock], dim=1)).squeeze(1)
