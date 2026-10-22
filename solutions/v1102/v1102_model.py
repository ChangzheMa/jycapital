"""原始字段结构编码、三频当天因果前缀与股票日共享预测。"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from v1102_schema import (FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs,
                         validate_backbone_config)

DAILY_BARS = {"bar1m": 240, "bar5m": 48, "bar30m": 8}


def validate_model_config(config):
    validate_backbone_config(config, "hierarchical_gru", (
        "intraday_layers", "interday_layers", "frequency_attention_heads", "frequency_fusion_layers",
        "frequency_ff_multiplier", "output_heads", "book_level_dim", "book_mixer_hidden_dim",
        "book_ff_multiplier", "non_book_dim", "expert_count", "expert_hidden_dim", "router_hidden_dim"))
    expected = {"normalization": "none", "pooling": "last_available_state",
                "recurrent_precision": "float32", "missing_history": "keep_slots_and_flags"}
    if any(config["model"].get(k) != v for k, v in expected.items()):
        raise ValueError("v1102 要求 FP32 单向递推、截点读取和保留历史缺失槽位")
    spec = config["model"]
    if spec["hidden_dim"] % spec["frequency_attention_heads"] or spec["output_heads"] != 2:
        raise ValueError("隐藏维度须整除注意力头数，输出为收益与排名两头")
    if spec["expert_count"] < 2:
        raise ValueError("多分支预测至少需要两个分支")
    scale = spec.get("residual_scale_initial")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("残差初始尺度必须在 (0, 1] 内")
    if spec.get("input_projection") != "raw_residual_structured_fields":
        raise ValueError("v1102 要求保留原线性投影并增加字段结构分支")
    scale = spec.get("input_residual_scale_initial")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("输入结构残差初始尺度必须在 (0, 1] 内")


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
        self.dense = nn.Linear(INPUT_CHANNELS, hidden)
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
        hidden = spec["hidden_dim"]
        self.daily_bars = DAILY_BARS[frequency]
        self.days = length // self.daily_bars
        self.project = StructuredFieldProjection(spec)
        self.dropout = nn.Dropout(spec["dropout"])
        self.intraday = nn.GRU(hidden, hidden, num_layers=spec["intraday_layers"], batch_first=True,
                              dropout=spec["dropout"] if spec["intraday_layers"] > 1 else 0)
        self.interday = nn.GRU(hidden, hidden, num_layers=spec["interday_layers"], batch_first=True,
                              dropout=spec["dropout"] if spec["interday_layers"] > 1 else 0)

    def forward(self, x, lengths):
        batch = x.shape[0]
        h = self.dropout(F.gelu(self.project(x.transpose(1, 2))))
        h = h.reshape(batch * self.days, self.daily_bars, -1)
        day_lengths = (lengths[:, None] - torch.arange(self.days, device=x.device)[None, :] * self.daily_bars).clamp(0, self.daily_bars)
        # FP32 状态递推兼容 CUDA GRU 并避免 BF16 长链状态误差；输入投影及预测头仍可 autocast。
        with torch.autocast(device_type=x.device.type, enabled=False):
            states, _ = self.intraday(h.float())
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
        vectors = [self.encoders[f](windows[f], lengths[f]) for f in FREQUENCIES]
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
            h = encoder.dropout(F.gelu(encoder.project(windows[frequency].transpose(1, 2))))
            with torch.autocast(device_type=device.type, enabled=False):
                days, bars = encoder.days, encoder.daily_bars
                states, _ = encoder.intraday(h.reshape(batch * days, bars, -1).float())
                states = states.reshape(batch, days, bars, -1)
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
                vectors.append(vector.reshape(8 * batch, -1))
        clocks = torch.eye(8, device=device, dtype=latest_clock.dtype).repeat_interleave(batch, dim=0)
        prediction, experts, logits = self.fuse_components(vectors, clocks)
        return (prediction.reshape(8, batch, 2).transpose(0, 1),
                experts.reshape(8, batch, -1, 2).transpose(0, 1),
                logits.reshape(8, batch, -1).transpose(0, 1))
