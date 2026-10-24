"""完整价格输入与连续关联状态，八时点共享因果计算。"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from v3001_schema import (FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs,
                         validate_backbone_config)
from v3001_price_memory import PriceHistoryMemory
from v3001_delta import DeltaBlock

DAILY_BARS = {"bar1m": 240, "bar5m": 48, "bar30m": 8}


def validate_model_config(config):
    keys = ('frequency_attention_heads', 'frequency_fusion_layers', 'frequency_ff_multiplier',
            'output_heads', 'book_level_dim', 'book_mixer_hidden_dim', 'book_ff_multiplier',
            'non_book_dim', 'expert_count', 'expert_hidden_dim', 'router_hidden_dim',
            'delta_layers', 'delta_heads', 'delta_key_dim', 'delta_value_dim',
            'delta_chunk_size', 'delta_conv_kernel', 'delta_ff_multiplier')
    validate_backbone_config(config, 'gated_delta', keys)
    spec = config['model']
    required = {'normalization': 'hidden_rms_only', 'pooling': 'last_causal_associative_state',
                'recurrent_precision': 'float32', 'missing_history': 'keep_slots_and_flags',
                'input_projection': 'raw_residual_structured_fields'}
    if any(spec.get(k) != v for k, v in required.items()):
        raise ValueError('关联状态的输入、精度或读出策略不一致')
    if spec['hidden_dim'] % spec['frequency_attention_heads'] or spec['output_heads'] != 2 or spec['expert_count'] < 2:
        raise ValueError('融合维度及输出头配置不一致')
    for key in ('residual_scale_initial', 'input_residual_scale_initial',
                'delta_initial_retention', 'delta_initial_write'):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(key + '须为(0,1)内有限数值')
    memory = spec.get('price_memory', {})
    for key in ('width', 'heads', 'distance_hidden_dim'):
        if type(memory.get(key)) is not int or memory[key] < 1:
            raise ValueError('历史读取参数必须为正整数: ' + key)
    if memory['width'] % memory['heads']:
        raise ValueError('历史读取宽度必须能被头数整除')
    expected = {'relative_position': 'learned_trading_bar_age', 'reference': 'latest_valid_closed_bar_at_query',
                'price_unit': '100_log_anchor_difference'}
    if any(memory.get(k) != v for k, v in expected.items()):
        raise ValueError('价格历史读取位置契约不一致')
    if not 0 < memory.get('residual_scale_initial', 0) <= 1:
        raise ValueError('历史读取尺度无效')


class StructuredFieldProjection(nn.Module):
    """同一 Bar 内按字段身份交互；不构造手工价量因子或跨时点统计量。"""

    def __init__(self, spec):
        super().__init__()
        hidden, level_dim = spec["hidden_dim"], spec["book_level_dim"]
        # 每档依次为 z 的六种盘口字段、c 的六种盘口字段；所有 60 个数值各出现一次。
        indices = [[base + group * 5 + level for base in (6, 42) for group in range(6)]
                   for level in range(5)]
        self.register_buffer("book_indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        self.register_buffer("non_book_indices", torch.tensor([*range(6), *range(36, 42), 72, 74]), persistent=False)
        self.dense = nn.Linear(75, hidden)
        self.level_project = nn.Linear(14, level_dim)
        self.level_position = nn.Parameter(torch.zeros(5, level_dim))
        self.level_mixer = nn.Sequential(nn.Linear(5, spec["book_mixer_hidden_dim"]), nn.GELU(),
                                         nn.Dropout(spec["dropout"]), nn.Linear(spec["book_mixer_hidden_dim"], 5))
        self.channel_mixer = nn.Sequential(nn.Linear(level_dim, level_dim * spec["book_ff_multiplier"]),
            nn.GELU(), nn.Dropout(spec["dropout"]), nn.Linear(level_dim * spec["book_ff_multiplier"], level_dim))
        self.non_book = nn.Sequential(nn.Linear(14, spec["non_book_dim"]), nn.GELU())
        self.combine = nn.Linear(5 * level_dim + spec["non_book_dim"] + 3, hidden)
        self.level_scale = nn.Parameter(torch.tensor(0.1))
        self.channel_scale = nn.Parameter(torch.tensor(0.1))
        self.output_scale = nn.Parameter(torch.tensor(float(spec["input_residual_scale_initial"])))

    def structured(self, x):
        book = x[..., self.book_indices]
        flags = x[..., 73:75].unsqueeze(-2).expand(*book.shape[:-1], 2)
        tokens = F.gelu(self.level_project(torch.cat([book, flags], dim=-1))) + self.level_position
        tokens = tokens + self.level_scale * self.level_mixer(tokens.transpose(-2, -1)).transpose(-2, -1)
        tokens = tokens + self.channel_scale * self.channel_mixer(tokens)
        non_book = self.non_book(x[..., self.non_book_indices])
        return self.combine(torch.cat([tokens.flatten(-2), non_book, x[..., 72:75]], dim=-1))

    def forward(self, x):
        return self.dense(x) + self.output_scale * self.structured(x)


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, frequency, length):
        super().__init__()
        self.daily_bars = DAILY_BARS[frequency]
        self.days = length // self.daily_bars
        self.length = length
        self.project = StructuredFieldProjection(spec)
        self.dropout = nn.Dropout(spec['dropout'])
        self.day_embedding = nn.Embedding(self.days, spec['hidden_dim'])
        self.slot_embedding = nn.Embedding(self.daily_bars, spec['hidden_dim'])
        nn.init.zeros_(self.day_embedding.weight)
        nn.init.zeros_(self.slot_embedding.weight)
        self.layers = nn.ModuleList([DeltaBlock(spec, self.daily_bars) for _ in range(spec['delta_layers'])])

    def sequence(self, raw, recurrent_reference=False):
        available = raw[:, 74, :] > 0
        position = torch.arange(self.length, device=raw.device)
        hidden = F.gelu(self.project(raw.transpose(1, 2)))
        hidden = hidden + self.day_embedding(position // self.daily_bars) + self.slot_embedding(position % self.daily_bars)
        hidden = self.dropout(hidden).masked_fill(~available[..., None], 0)
        for layer in self.layers:
            hidden = layer(hidden, available, recurrent_reference=recurrent_reference)
        return hidden, available

    def forward(self, raw, lengths, recurrent_reference=False):
        # Q个读出共享一条连续历史状态；缺失整根Bar不更新，读取此前最后有效位置。
        hidden, available = self.sequence(raw, recurrent_reference)
        one = lengths.ndim == 1
        lengths = lengths[:, None] if one else lengths
        positions = torch.arange(self.length, device=raw.device)[None, :]
        last = torch.where(available, positions, -1).cummax(-1).values
        indices = last.gather(1, (lengths - 1).clamp_min(0))
        selected = hidden.gather(1, indices.clamp_min(0)[..., None].expand(-1, -1, hidden.shape[-1]))
        selected = selected.masked_fill(((lengths == 0) | (indices < 0))[..., None], 0)
        return selected[:, 0] if one else selected


class FrequencyFusion(nn.Module):
    """仅在同一股票、同一预测时点的三个频率之间交互。"""

    def __init__(self, spec):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.heads = spec["frequency_attention_heads"]
        self.width = hidden // self.heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.projection = nn.Linear(hidden, hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * spec["frequency_ff_multiplier"]),
                                nn.GELU(), nn.Dropout(spec["dropout"]),
                                nn.Linear(hidden * spec["frequency_ff_multiplier"], hidden))
        self.dropout = nn.Dropout(spec["dropout"])
        self.attention_scale = nn.Parameter(torch.tensor(float(spec["residual_scale_initial"])))
        self.ff_scale = nn.Parameter(torch.tensor(float(spec["residual_scale_initial"])))

    def forward(self, tokens):
        batch, length, hidden = tokens.shape
        q, k, v = self.qkv(tokens).reshape(batch, length, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        weights = torch.softmax((q @ k.transpose(-1, -2)).float() / math.sqrt(self.width), dim=-1).to(v.dtype)
        value = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, length, hidden)
        tokens = tokens + self.attention_scale * self.dropout(self.projection(value))
        return tokens + self.ff_scale * self.dropout(self.ff(tokens))


class MultiFrequencyDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_model_config(config)
        spec = config["model"]
        self.window_bars = dict(config["data"]["window_bars"])
        self.encoders = nn.ModuleDict({f: FrequencyEncoder(spec, f, self.window_bars[f]) for f in FREQUENCIES})
        self.frequency_embedding = nn.Parameter(torch.zeros(len(FREQUENCIES), spec["hidden_dim"]))
        self.frequency_fusion = nn.Sequential(*[FrequencyFusion(spec) for _ in range(spec["frequency_fusion_layers"])])
        self.shared = nn.Sequential(
            nn.Linear(len(FREQUENCIES) * spec["hidden_dim"] + spec["prediction_clock_dim"], spec["fusion_hidden_dim"]),
            nn.GELU(), nn.Dropout(spec["dropout"]))
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(spec["fusion_hidden_dim"], spec["expert_hidden_dim"]), nn.GELU(),
            nn.Dropout(spec["dropout"]), nn.Linear(spec["expert_hidden_dim"], spec["output_heads"]))
            for _ in range(spec["expert_count"])])
        self.router = nn.Sequential(nn.Linear(spec["fusion_hidden_dim"], spec["router_hidden_dim"]),
                                    nn.GELU(), nn.Linear(spec["router_hidden_dim"], spec["expert_count"]))
        self.price_memories = nn.ModuleDict({f: PriceHistoryMemory(spec['hidden_dim'], self.window_bars[f],
                                              spec['price_memory'], spec['dropout']) for f in FREQUENCIES})

    def fuse(self, vectors, clock):
        return self.fuse_components(vectors, clock)[0]

    def fuse_components(self, vectors, clock):
        tokens = torch.stack(vectors, dim=1) + self.frequency_embedding
        tokens = self.frequency_fusion(tokens)
        shared = self.shared(torch.cat([tokens.flatten(1), clock], dim=1))
        experts = torch.stack([expert(shared) for expert in self.experts], dim=1).float()
        logits = self.router(shared).float()
        probabilities = torch.softmax(logits, dim=-1)
        return (experts * probabilities[..., None]).sum(dim=1), experts, logits

    def forward_heads(self, windows, clock):
        """逐样本参考路径；不使用共享前缀，便于逻辑和小样本对照。"""
        windows, lengths = prepare_model_inputs(windows, clock, self.window_bars)
        vectors = []
        for f in FREQUENCIES:
            raw = windows[f][:, :75, :]
            vector = self.encoders[f](raw, lengths[f])
            vectors.append(self.price_memories[f].single(vector, raw, windows[f][:, 75, :],
                                                        windows[f][:, 76, :] > 0, lengths[f]))
        return self.fuse(vectors, clock)

    def forward(self, windows, clock):
        return self.forward_heads(windows, clock)[:, 0]

    def forward_group(self, windows, latest_clock):
        """股票日共享推理，返回混合后的收益／排名双头。"""
        return self.forward_group_components(windows, latest_clock)[0]

    def forward_group_components(self, windows, latest_clock):
        windows, _ = prepare_model_inputs(windows, latest_clock, self.window_bars)
        batch, device = latest_clock.shape[0], latest_clock.device
        vectors = []
        for frequency in FREQUENCIES:
            encoder = self.encoders[frequency]
            raw = windows[frequency][:, :75, :]
            lengths = ((encoder.days - 1) * encoder.daily_bars
                       + torch.arange(8, device=device) * (encoder.daily_bars // 8))[None, :].expand(batch, -1)
            vector = encoder(raw, lengths)
            vector = self.price_memories[frequency](vector, raw, windows[frequency][:, 75, :],
                                                    windows[frequency][:, 76, :] > 0, lengths)
            vectors.append(vector.transpose(0, 1).reshape(8 * batch, -1))
        clocks = torch.eye(8, device=device, dtype=latest_clock.dtype).repeat_interleave(batch, dim=0)
        prediction, experts, logits = self.fuse_components(vectors, clocks)
        return (prediction.reshape(8, batch, 2).transpose(0, 1),
                experts.reshape(8, batch, -1, 2).transpose(0, 1),
                logits.reshape(8, batch, -1).transpose(0, 1))
