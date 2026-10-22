"""已完成交易日的可学习历史压缩与短期条件读出。"""
import math

import torch
from torch import nn

from long_history import validate_long_config


class HistoryAttention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        width = spec["day_dim"]
        self.heads, self.head_width = spec["attention_heads"], width // spec["attention_heads"]
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * spec["ff_multiplier"]), nn.GELU(),
                                          nn.Dropout(spec["dropout"]), nn.Linear(width * spec["ff_multiplier"], width))
        self.dropout = nn.Dropout(spec["dropout"])

    def forward(self, tokens, valid):
        batch, length, width = tokens.shape
        q, k, v = self.qkv(self.norm1(tokens)).reshape(batch, length, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        safe = valid.clone()
        safe[~valid.any(dim=1), 0] = True
        scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.head_width)
        scores = scores.masked_fill(~safe[:, None, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1).to(v.dtype)
        update = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.dropout(self.projection(update))
        tokens = tokens + self.dropout(self.feedforward(self.norm2(tokens)))
        return torch.where(valid[..., None], tokens, 0)


class LongHistoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_long_config(config)
        spec = config["long_context"]
        self.days, self.patch_days = spec["previous_days"], spec["patch_days"]
        width = spec["day_dim"]
        self.heads, self.head_width = spec["attention_heads"], width // spec["attention_heads"]
        self.day_projection = nn.Sequential(nn.Linear(8 * 75, width), nn.GELU(), nn.Dropout(spec["dropout"]))
        self.patch_projection = nn.Linear(width * self.patch_days, width)
        self.positions = nn.Parameter(torch.empty(1, self.days // self.patch_days, width))
        nn.init.normal_(self.positions, std=.02)
        self.blocks = nn.ModuleList([HistoryAttention(spec) for _ in range(spec["encoder_layers"])])
        self.memory_norm = nn.LayerNorm(width)
        fusion = config["model"]["fusion_hidden_dim"]
        self.query_norm = nn.LayerNorm(fusion)
        self.query = nn.Linear(fusion, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, fusion)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.scale = nn.Parameter(torch.tensor(float(spec["residual_scale_initial"])))
        self.dropout = nn.Dropout(spec["dropout"])

    def encode(self, values):
        if values.ndim != 3 or values.shape[1:] != (75, self.days * 8):
            raise ValueError("长期分支输入必须为固定历史日的75通道8根Bar")
        batch = values.shape[0]
        daily = values.transpose(1, 2).reshape(batch, self.days, 8, 75)
        bar_valid = daily[..., 74] > 0
        daily = torch.where(bar_valid[..., None], daily, 0)
        day_valid = bar_valid.any(dim=2)
        tokens = self.day_projection(daily.flatten(2))
        tokens = torch.where(day_valid[..., None], tokens, 0)
        valid = day_valid.reshape(batch, -1, self.patch_days).any(dim=2)
        tokens = self.patch_projection(tokens.reshape(batch, self.days // self.patch_days, -1)) + self.positions
        tokens = torch.where(valid[..., None], tokens, 0)
        for block in self.blocks:
            tokens = block(tokens, valid)
        return self.memory_norm(tokens), valid

    def read(self, queries, encoded):
        """queries为[股票日,时点,融合宽度]，一次投影历史供全部时点读取。"""
        memory, valid = encoded
        batch, length, width = memory.shape
        count = queries.shape[1]
        q = self.query(self.query_norm(queries)).reshape(batch, count, self.heads, self.head_width).transpose(1, 2)
        k, v = self.key_value(memory).reshape(batch, length, 2, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        safe = valid.clone()
        safe[~valid.any(dim=1), 0] = True
        scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.head_width)
        scores = scores.masked_fill(~safe[:, None, None, :], -torch.inf)
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        value = (self.dropout(attention) @ v).transpose(1, 2).reshape(batch, count, width)
        result = self.scale * self.dropout(self.output(value))
        return torch.where(valid.any(dim=1)[:, None, None], result, 0)
