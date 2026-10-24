"""每档保持独立因果状态；同Bar空间交互后再次沿时间递推。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_field_grid(spec):
    grid = spec.get('field_grid', {})
    expected = {'node_count': 6, 'temporal_stages': 2, 'spatial_blocks': 1,
                'normalization': 'none', 'day_boundary': 'reset'}
    if any(grid.get(k) != v for k, v in expected.items()):
        raise ValueError('字段网格要求六节点、时间-空间-时间结构和日边界重置')
    for key in ('width', 'attention_heads', 'ff_multiplier'):
        if type(grid.get(key)) is not int or grid[key] < 1:
            raise ValueError('字段网格整数配置非法: ' + key)
    if grid['width'] % grid['attention_heads']:
        raise ValueError('字段网格宽度须能被注意力头数整除')
    for key in ('grid_scale_initial', 'spatial_residual_scale_initial', 'node_embedding_std'):
        value = grid.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('字段网格初始尺度须在(0,1]内: ' + key)
    if spec.get('intraday_encoder') != 'time_space_time_field_grid':
        raise ValueError('字段网格必须作为日内编码器')


class SynchronousNodeMixing(nn.Module):
    """输入[B,T,6,W]；每个Bar的空间交互独立，不归一化时间或股票轴。"""

    def __init__(self, grid, dropout):
        super().__init__()
        width = grid['width']
        self.heads = grid['attention_heads']
        self.head_width = width // self.heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, width * grid['ff_multiplier']), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(width * grid['ff_multiplier'], width))
        self.dropout = nn.Dropout(dropout)
        self.attention_scale = nn.Parameter(torch.tensor(float(grid['spatial_residual_scale_initial'])))
        self.ff_scale = nn.Parameter(torch.tensor(float(grid['spatial_residual_scale_initial'])))

    def forward(self, tokens):
        batch, length, nodes, width = tokens.shape
        flat = tokens.reshape(batch * length, nodes, width)
        q, k, v = self.qkv(flat).reshape(batch * length, nodes, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        attention = torch.softmax((q @ k.transpose(-1, -2)).float() / math.sqrt(self.head_width), dim=-1).to(v.dtype)
        value = (self.dropout(attention) @ v).transpose(1, 2).reshape(batch * length, nodes, width)
        flat = flat + self.attention_scale * self.dropout(self.projection(value))
        flat = flat + self.ff_scale * self.dropout(self.ff(flat))
        return flat.reshape(batch, length, nodes, width)


class IntradayFieldGrid(nn.Module):
    """每个调用样本是一日，节点共享GRU参数但不共享隐状态。"""

    def __init__(self, spec):
        super().__init__()
        validate_field_grid(spec)
        grid = spec['field_grid']
        width = grid['width']
        indices = [[base + group * 5 + level for base in (6, 42) for group in range(6)]
                   for level in range(5)]
        self.register_buffer('book_indices', torch.tensor(indices, dtype=torch.long), persistent=False)
        self.register_buffer('non_book_indices', torch.tensor([*range(6), *range(36, 42), 72, 74]), persistent=False)
        self.book_project = nn.Linear(14, width)
        self.non_book_project = nn.Linear(14, width)
        self.node_embedding = nn.Parameter(torch.empty(6, width))
        nn.init.normal_(self.node_embedding, std=grid['node_embedding_std'])
        self.temporal_first = nn.GRU(width, width, batch_first=True)
        self.spatial = SynchronousNodeMixing(grid, spec['dropout'])
        self.temporal_second = nn.GRU(width, width, batch_first=True)
        self.readout = nn.Linear(6 * width, spec['hidden_dim'])
        self.raw_skip = nn.Linear(75, spec['hidden_dim'])
        self.grid_scale = nn.Parameter(torch.tensor(float(grid['grid_scale_initial'])))
        self.dropout = nn.Dropout(spec['dropout'])

    def field_tokens(self, raw):
        book = raw[..., self.book_indices]
        flags = raw[..., 73:75].unsqueeze(-2).expand(*book.shape[:-1], 2)
        book = F.gelu(self.book_project(torch.cat([book, flags], dim=-1)))
        non_book = F.gelu(self.non_book_project(raw[..., self.non_book_indices])).unsqueeze(-2)
        return torch.cat([book, non_book], dim=-2) + self.node_embedding

    @staticmethod
    def temporal_pass(tokens, recurrent):
        batch, length, nodes, width = tokens.shape
        sequences = tokens.permute(0, 2, 1, 3).reshape(batch * nodes, length, width)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            states, _ = recurrent(sequences.float())
        return states.reshape(batch, nodes, length, width).permute(0, 2, 1, 3)

    def forward(self, raw):
        if raw.ndim != 3 or raw.shape[-1] != 75:
            raise ValueError('字段网格输入须为[股票日,日内Bar,75]')
        tokens = self.dropout(self.field_tokens(raw))
        first = self.temporal_pass(tokens, self.temporal_first)
        mixed = self.spatial(first)
        second = self.temporal_pass(self.dropout(mixed), self.temporal_second)
        return (self.raw_skip(raw) + self.grid_scale * self.readout(second.flatten(-2))).float()
