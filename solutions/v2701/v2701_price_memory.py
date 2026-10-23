"""用当前因果状态查询完整原始历史，并学习相对价格位置；不接收收益标签。"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PriceHistoryMemory(nn.Module):
    """原 75 通道保持，另接收 FP32 对数中心及有效标志。"""

    def __init__(self, state_width, window_length, specification, dropout):
        super().__init__()
        self.state_width = int(state_width)
        self.window_length = int(window_length)
        self.width = int(specification['width'])
        self.heads = int(specification['heads'])
        if self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError('历史读取宽度须能被头数整除')
        if self.window_length < 1:
            raise ValueError('历史窗口必须为正')
        self.head_width = self.width // self.heads
        self.query = nn.Linear(self.state_width, self.width)
        self.raw_projection = nn.Sequential(nn.Linear(75, self.width), nn.GELU(), nn.Dropout(dropout))
        self.key = nn.Linear(self.width, self.width)
        self.value = nn.Linear(self.width, self.width)
        self.age_bias = nn.Embedding(self.window_length, self.heads)
        nn.init.zeros_(self.age_bias.weight)
        self.price_bias = nn.Sequential(nn.Linear(2, specification['distance_hidden_dim']),
                                        nn.GELU(), nn.Linear(specification['distance_hidden_dim'], self.heads))
        nn.init.zeros_(self.price_bias[-1].weight)
        nn.init.zeros_(self.price_bias[-1].bias)
        self.output = nn.Linear(self.width + 2 * self.heads, self.state_width)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(specification['residual_scale_initial'])))

    def forward(self, states, raw, log_anchor, anchor_available, lengths, reference=None):
        """states[B,Q,H]，raw[B,75,L]；长度是每个查询允许读取的位置数。"""
        if states.ndim != 3 or raw.ndim != 3:
            raise ValueError('历史读取要求批量、查询／字段、位置三个维度')
        batch, queries, hidden = states.shape
        size = raw.shape[-1]
        if hidden != self.state_width or raw.shape != (batch, 75, self.window_length):
            raise ValueError('历史内容或状态维度与配置不符')
        if log_anchor.shape != (batch, size) or anchor_available.shape != (batch, size):
            raise ValueError('价格中心与可用性形状不符')
        if lengths.shape != (batch, queries):
            raise ValueError('每个查询必须有独立因果长度')
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError('因果长度必须为整数')
        if torch.any(lengths < 0) or torch.any(lengths > size):
            raise ValueError('因果长度超出窗口')
        lengths = lengths.long()
        if batch == 0:
            return states

        # 先计算每根 Bar 的独立表示，因此 key/value 可共享，未来内容不能影响过去表示。
        projected = self.raw_projection(raw.transpose(1, 2))
        if reference is None:
            raise ValueError('v2701历史内容必须包含成交与日级参考输入')
        projected = projected + self.reference_projection(reference)
        keys = self.key(projected).reshape(batch, size, self.heads, self.head_width)
        values = self.value(projected).reshape(batch, size, self.heads, self.head_width)
        query = self.query(states).reshape(batch, queries, self.heads, self.head_width)
        positions = torch.arange(size, device=raw.device)
        available = anchor_available.bool() & torch.isfinite(log_anchor) & (raw[:, 74, :] > 0)
        safe_anchor = torch.where(available, log_anchor, torch.zeros_like(log_anchor)).float()

        # 每个截止位置读取此前最后有效中心；不使用窗口最后值替代早期查询中心。
        prior = torch.where(available, positions[None, :], -1).cummax(dim=-1).values
        query_index = prior.gather(1, (lengths - 1).clamp_min(0))
        query_available = (lengths > 0) & (query_index >= 0)
        query_anchor = safe_anchor.gather(1, query_index.clamp_min(0))
        pair_available = query_available[:, :, None] & available[:, None, :]
        distance = 100. * (safe_anchor[:, None, :] - query_anchor[:, :, None])
        distance = torch.where(pair_available, distance, torch.zeros_like(distance))
        age = (lengths[:, :, None] - 1 - positions[None, None, :]).clamp(0, size - 1)
        causal = positions[None, None, :] < lengths[:, :, None]
        visible = causal & (raw[:, 74, :] > 0)[:, None, :]

        # 距离与价格有效性作为可学习的相对位置输入；价格差和加权累计保留 FP32。
        price_input = torch.stack((distance, pair_available.float()), dim=-1)
        bias = self.price_bias(price_input).permute(0, 1, 3, 2)
        bias = bias + self.age_bias(age).permute(0, 1, 3, 2)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            scores = torch.einsum('bqhd,blhd->bqhl', query.float(), keys.float()) / math.sqrt(self.head_width)
            scores = scores + bias.float()
            nonempty = visible.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~visible[:, :, None, :], float('-inf'))
            scores = torch.where(nonempty[:, :, None, :], scores, torch.zeros_like(scores))
            attention = F.softmax(scores, dim=-1)
            attention = attention * visible[:, :, None, :]
            content = torch.einsum('bqhl,blhd->bqhd', attention, values.float()).flatten(-2)
            price_position = (attention * distance[:, :, None, :]).sum(dim=-1)
            price_mass = (attention * pair_available[:, :, None, :]).sum(dim=-1)
            combined = torch.cat((content, price_position, price_mass), dim=-1)
        update = self.dropout(self.output(combined))
        # 全空上下文没有额外偏置；保留原状态，不引入伪历史。
        update = update * nonempty
        return states + self.residual_scale * update

    def single(self, states, raw, log_anchor, anchor_available, lengths, reference=None):
        """单查询参考接口，主模型逐时点路径复用相同声明的数学运算。"""
        return self.forward(states[:, None, :], raw, log_anchor, anchor_available, lengths[:, None], reference)[:, 0]
