"""v9 三路固定交易日窗口网络：75 通道、展平投影和预测时刻 one-hot。"""

import torch
from torch import nn
from torch.nn import functional as F

from v9_schema import FREQUENCIES, INPUT_CHANNELS, validate_input_config


def validate_model_config(config):
    validate_input_config(config)
    spec = config["model"]
    if set(spec["blocks"]) != set(FREQUENCIES) or set(spec["kernel_sizes"]) != set(FREQUENCIES):
        raise ValueError("v9 只包含 1、5、30 分钟三个分支")
    expected = {"hidden_validity_mask": False, "pooling": "flatten",
                "padding": "symmetric_replicate", "dilation": 1}
    if any(spec.get(key) != value for key, value in expected.items()) or "kernel_size" in spec:
        raise ValueError("v9 要求无隐藏掩码、展平投影、对称复制填充及膨胀率 1")
    for frequency in FREQUENCIES:
        blocks = spec["blocks"][frequency]
        length = config["data"]["window_bars"][frequency]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (blocks, length)):
            raise ValueError("残差块数量和窗口长度必须为正整数")
        if spec["kernel_sizes"][frequency] != [3 + 2 * i for i in range(blocks)]:
            raise ValueError("v9 各分支卷积核必须按残差块使用宽度 3、5、7、9")


class SymmetricBlock(nn.Module):
    def __init__(self, hidden, kernel, dropout):
        super().__init__()
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("对称复制卷积要求正奇数核宽")
        self.padding = (kernel - 1) // 2
        self.convolutions = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel, dilation=1) for _ in range(2)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        for conv in self.convolutions:
            x = conv(F.pad(x, (self.padding, self.padding), mode="replicate"))
            x = self.dropout(F.gelu(x))
        return x + residual


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, kernels, window_bars):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.project = nn.Conv1d(INPUT_CHANNELS, hidden, 1)
        self.blocks = nn.ModuleList([
            SymmetricBlock(hidden, kernel, spec["tcn_dropout"]) for kernel in kernels
        ])
        self.window_bars = window_bars
        self.pool = nn.Linear(hidden * window_bars, hidden)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (INPUT_CHANNELS, self.window_bars):
            raise ValueError("分支输入必须为 [batch, 75, 配置窗口长度]")
        h = F.gelu(self.project(x))
        for block in self.blocks:
            h = block(h)
        return F.gelu(self.pool(h.flatten(start_dim=1)))


class MultiFrequencyTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_model_config(config)
        spec = config["model"]
        self.encoders = nn.ModuleDict({
            frequency: FrequencyEncoder(spec, spec["kernel_sizes"][frequency],
                                        config["data"]["window_bars"][frequency])
            for frequency in FREQUENCIES
        })
        self.clock_dim = spec["prediction_clock_dim"]
        self.head = nn.Sequential(
            nn.Linear(len(FREQUENCIES) * spec["hidden_dim"] + self.clock_dim, spec["fusion_hidden_dim"]),
            nn.GELU(), nn.Dropout(spec["dropout"]), nn.Linear(spec["fusion_hidden_dim"], 1),
        )

    def forward(self, windows, clock):
        if clock.ndim != 2 or clock.shape[1] != self.clock_dim:
            raise ValueError("v9 预测时刻必须为八维 one-hot")
        vectors = [self.encoders[f](windows[f]) for f in FREQUENCIES]
        return self.head(torch.cat([*vectors, clock], dim=1)).squeeze(1)
