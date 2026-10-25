"""原始字段结构编码、三频当天因果前缀与股票日共享预测。"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from v3201_schema import (FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs,
                         validate_backbone_config)
from v3201_price_memory import PriceHistoryMemory
from v3201_field_grid import IntradayFieldGrid, validate_field_grid

DAILY_BARS = {"bar1m": 240, "bar5m": 48, "bar30m": 8}


def validate_model_config(config):
    validate_backbone_config(config, "hierarchical_gru", (
        "interday_layers", "frequency_attention_heads", "frequency_fusion_layers",
        "frequency_ff_multiplier", "output_heads", "expert_count", "expert_hidden_dim", "router_hidden_dim"))
    expected = {"normalization": "none", "pooling": "last_available_state",
                "recurrent_precision": "float32", "missing_history": "keep_slots_and_flags"}
    if any(config["model"].get(k) != v for k, v in expected.items()):
        raise ValueError("v3201 要求 FP32 单向递推、截点读取和保留历史缺失槽位")
    spec = config["model"]
    if spec["hidden_dim"] % spec["frequency_attention_heads"] or spec["output_heads"] != 2:
        raise ValueError("隐藏维度须整除注意力头数，输出为收益与排名两头")
    if spec["expert_count"] < 2:
        raise ValueError("多分支预测至少需要两个分支")
    scale = spec.get("residual_scale_initial")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("残差初始尺度必须在 (0, 1] 内")
    if spec.get('input_projection') != 'persistent_field_grid_with_raw_skip':
        raise ValueError('v3201要求保持字段节点的日内编码和原始线性路径')
    validate_field_grid(spec)
    memory = spec.get('price_memory', {})
    for key in ('width', 'heads', 'distance_hidden_dim'):
        if type(memory.get(key)) is not int or memory[key] < 1:
            raise ValueError('历史读取参数必须为正整数: ' + key)
    if memory['width'] % memory['heads']:
        raise ValueError('历史读取宽度必须能被头数整除')
    if memory.get('relative_position') != 'learned_trading_bar_age' or memory.get('reference') != 'latest_valid_closed_bar_at_query' or memory.get('price_unit') != '100_log_anchor_difference':
        raise ValueError('价格历史读取位置契约不一致')
    scale = memory.get('residual_scale_initial')
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('历史读取残差尺度必须在 (0,1] 内')


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, frequency, length):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.daily_bars = DAILY_BARS[frequency]
        self.days = length // self.daily_bars
        self.intraday = IntradayFieldGrid(spec)
        self.dropout = nn.Dropout(spec["dropout"])
        self.interday = nn.GRU(hidden, hidden, num_layers=spec["interday_layers"], batch_first=True,
                              dropout=spec["dropout"] if spec["interday_layers"] > 1 else 0)

    def intraday_states(self, x):
        batch = x.shape[0]
        raw = x.transpose(1, 2).reshape(batch * self.days, self.daily_bars, 75)
        return self.intraday(raw).reshape(batch, self.days, self.daily_bars, -1)

    def forward(self, x, lengths):
        batch = x.shape[0]
        states = self.intraday_states(x).reshape(batch * self.days, self.daily_bars, -1)
        day_lengths = (lengths[:, None] - torch.arange(self.days, device=x.device)[None, :] * self.daily_bars).clamp(0, self.daily_bars)
        # FP32 状态递推兼容 CUDA GRU 并避免 BF16 长链状态误差；输入投影及预测头仍可 autocast。
        with torch.autocast(device_type=x.device.type, enabled=False):
            last = (day_lengths.flatten() - 1).clamp_min(0)
            daily = states[torch.arange(batch * self.days, device=x.device), last]
            daily = daily.reshape(batch, self.days, -1)
            daily = daily.masked_fill((day_lengths == 0)[:, :, None], 0)
            states, _ = self.interday(self.dropout(daily))
            # 09:30 今日尚无 Bar，直接读取昨日状态；不让未来零槽位改变输出。
            last_day = (lengths - 1) // self.daily_bars
            return states[torch.arange(batch, device=x.device), last_day]


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


class MultiFrequencyGRU(nn.Module):
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
        """返回混合双头、各分支分数和路由 logits；标签只在外部损失中使用。"""
        windows, _ = prepare_model_inputs(windows, latest_clock, self.window_bars)
        batch, device = latest_clock.shape[0], latest_clock.device
        vectors = []
        for frequency in FREQUENCIES:
            encoder = self.encoders[frequency]
            raw = windows[frequency][:, :75, :]
            states = encoder.intraday_states(raw)
            with torch.autocast(device_type=device.type, enabled=False):
                days, bars = encoder.days, encoder.daily_bars
                # 各频率的历史日先跨日编码一次，八个时点从同一历史状态分别接入今日前缀。
                previous, history_state = encoder.interday(encoder.dropout(states[:, :-1, -1]))
                last = (torch.arange(8, device=device) * (bars // 8) - 1).clamp_min(0)
                current = states[:, -1, last].transpose(0, 1).clone()
                current[0] = 0
                updated, _ = encoder.interday(encoder.dropout(current).reshape(8 * batch, 1, -1),
                                               history_state.repeat(1, 8, 1))
                updated = updated[:, 0].reshape(8, batch, -1)
                available_today = (torch.arange(8, device=device) > 0)[:, None, None]
                vector = torch.where(available_today, updated, previous[:, -1].unsqueeze(0))
            query_lengths = ((days - 1) * bars + torch.arange(8, device=device) * (bars // 8))[None, :].expand(batch, -1)
            vector = self.price_memories[frequency](vector.transpose(0, 1), raw,
                        windows[frequency][:, 75, :], windows[frequency][:, 76, :] > 0, query_lengths)
            vectors.append(vector.transpose(0, 1).reshape(8 * batch, -1))
        clocks = torch.eye(8, device=device, dtype=latest_clock.dtype).repeat_interleave(batch, dim=0)
        prediction, experts, logits = self.fuse_components(vectors, clocks)
        return (prediction.reshape(8, batch, 2).transpose(0, 1),
                experts.reshape(8, batch, -1, 2).transpose(0, 1),
                logits.reshape(8, batch, -1).transpose(0, 1))
