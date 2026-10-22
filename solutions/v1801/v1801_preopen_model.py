"""原始盘前逐 Bar 编码、跨日状态与预测时点条件读出。"""
import math

import torch
from torch import nn

from preopen_context import BARS, validate_preopen_config


class PhaseAttention(nn.Module):
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


class PreopenEncoder(nn.Module):
    def __init__(self, config, projection_factory):
        super().__init__()
        validate_preopen_config(config)
        spec = config["preopen_context"]
        self.days = spec["previous_days"] + 1
        width = spec["day_dim"]
        self.heads, self.head_width = spec["attention_heads"], width // spec["attention_heads"]
        self.project = projection_factory({**config["model"], "hidden_dim": width, "dropout": spec["dropout"]})
        self.bar_positions = nn.Parameter(torch.empty(1, BARS, width))
        self.day_positions = nn.Parameter(torch.empty(1, self.days, width))
        nn.init.normal_(self.bar_positions, std=.02)
        nn.init.normal_(self.day_positions, std=.02)
        self.intraday = nn.ModuleList([PhaseAttention(spec) for _ in range(spec["intraday_layers"])])
        self.daily = nn.Sequential(nn.Linear(BARS * width, width), nn.GELU(), nn.Dropout(spec["dropout"]))
        self.history = nn.ModuleList([PhaseAttention(spec) for _ in range(spec["encoder_layers"])])
        self.context_scale = nn.Parameter(torch.tensor(.1))
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
        if values.ndim != 3 or values.shape[1:] != (75, self.days * BARS):
            raise ValueError("盘前输入必须包含当前及前置交易日的75通道10根Bar")
        batch = values.shape[0]
        daily = values.transpose(1, 2).reshape(batch, self.days, BARS, 75)
        bar_valid = daily[..., 74] > 0
        daily = torch.where(bar_valid[..., None], daily, 0)
        tokens = self.dropout(torch.nn.functional.gelu(self.project(daily.reshape(batch * self.days, BARS, 75))))
        tokens = tokens + self.bar_positions
        valid_flat = bar_valid.reshape(batch * self.days, BARS)
        tokens = torch.where(valid_flat[..., None], tokens, 0)
        for block in self.intraday:
            tokens = block(tokens, valid_flat)
        tokens = tokens.reshape(batch, self.days, BARS, -1)
        day_valid = bar_valid.any(dim=2)
        days = self.daily(tokens.flatten(2)) + self.day_positions
        days = torch.where(day_valid[..., None], days, 0)
        for block in self.history:
            days = block(days, day_valid)
        # 全部盘前槽位均早于09:30；保留当前日细粒度序列供八时点分别读取。
        current = tokens[:, -1] + self.context_scale * days[:, -1, None, :]
        memory = torch.cat([days[:, :-1], current], dim=1)
        valid = torch.cat([day_valid[:, :-1], bar_valid[:, -1]], dim=1)
        return self.memory_norm(memory), valid

    def read(self, queries, encoded):
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
