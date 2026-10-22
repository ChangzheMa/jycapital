"""原始字段结构编码、三频当天因果前缀与股票日共享预测。"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from v2101_schema import (FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs,
                         validate_backbone_config)

DAILY_BARS = {"bar1m": 240, "bar5m": 48, "bar30m": 8}


def validate_model_config(config):
    validate_backbone_config(config, "hierarchical_gru", (
        "intraday_layers", "interday_layers", "frequency_attention_heads", "frequency_fusion_layers",
        "frequency_ff_multiplier", "output_heads", "book_level_dim", "book_mixer_hidden_dim",
        "book_ff_multiplier", "non_book_dim", "expert_count", "task_hidden_dim", "task_feature_dim",
        "task_router_hidden_dim", "quantity_head_hidden_dim", "return_head_hidden_dim"))
    expected = {"normalization": "none", "pooling": "last_available_state",
                "recurrent_precision": "float32", "missing_history": "keep_slots_and_flags"}
    if any(config["model"].get(k) != v for k, v in expected.items()):
        raise ValueError("v2101 要求 FP32 单向递推、截点读取和保留历史缺失槽位")
    spec = config["model"]
    if spec["hidden_dim"] % spec["frequency_attention_heads"] or spec["output_heads"] != 2:
        raise ValueError("隐藏维度须整除注意力头数，输出为收益与排名两头")
    if spec["expert_count"] < 2:
        raise ValueError("多分支预测至少需要两个分支")
    if spec.get("output_mechanism") != "shared_private_liquidity_conditioned_return" or spec.get("quantity_fields") != 22:
        raise ValueError("v2101 要求共享／专属任务特征和22个数量变化输出")
    scale = spec.get("residual_scale_initial")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("残差初始尺度必须在 (0, 1] 内")
    if spec.get("input_projection") != "raw_residual_structured_fields":
        raise ValueError("v2101 要求保留原线性投影并增加字段结构分支")
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
        def expert():
            return nn.Sequential(nn.Linear(spec["fusion_hidden_dim"], spec["task_hidden_dim"]), nn.GELU(),
                nn.Dropout(spec["dropout"]), nn.Linear(spec["task_hidden_dim"], spec["task_feature_dim"]), nn.GELU())
        def gate():
            return nn.Sequential(nn.Linear(spec["fusion_hidden_dim"], spec["task_router_hidden_dim"]), nn.GELU(),
                                 nn.Linear(spec["task_router_hidden_dim"], spec["expert_count"] + 1))
        self.shared_experts = nn.ModuleList([expert() for _ in range(spec["expert_count"])])
        self.return_private, self.quantity_private = expert(), expert()
        self.return_gate, self.quantity_gate = gate(), gate()
        self.quantity_head = nn.Sequential(nn.Linear(spec["task_feature_dim"], spec["quantity_head_hidden_dim"]),
                                          nn.GELU(), nn.Linear(spec["quantity_head_hidden_dim"], 22))
        self.return_head = nn.Sequential(nn.Linear(2 * spec["task_feature_dim"] + 22, spec["return_head_hidden_dim"]),
                                        nn.GELU(), nn.Linear(spec["return_head_hidden_dim"], 2))

    def fuse(self, vectors, clock):
        return self.fuse_tasks(vectors, clock)["prediction"]

    def fuse_tasks(self, vectors, clock):
        tokens = torch.stack(vectors, dim=1) + self.frequency_embedding
        tokens = self.frequency_fusion(tokens)
        shared = self.shared(torch.cat([tokens.flatten(1), clock], dim=1))
        experts = torch.stack([expert(shared) for expert in self.shared_experts], dim=1).float()
        return_gate = torch.softmax(self.return_gate(shared).float(), dim=-1)
        quantity_gate = torch.softmax(self.quantity_gate(shared).float(), dim=-1)
        return_experts = torch.cat([experts, self.return_private(shared).float()[:, None]], dim=1)
        quantity_experts = torch.cat([experts, self.quantity_private(shared).float()[:, None]], dim=1)
        return_feature = (return_experts * return_gate[..., None]).sum(dim=1)
        quantity_feature = (quantity_experts * quantity_gate[..., None]).sum(dim=1)
        delta = self.quantity_head(quantity_feature).float()
        prediction = self.return_head(torch.cat([return_feature, quantity_feature, delta], dim=-1)).float()
        # 仅训练记录作用幅度；复用同一次特征且读出不含Dropout，差异不混入随机掩码。
        without_quantity = self.return_head(torch.cat([return_feature, torch.zeros_like(quantity_feature),
                                                       torch.zeros_like(delta)], dim=-1)).float() if self.training else prediction
        return {"prediction": prediction, "quantity_delta": delta,
                "return_gate": return_gate, "quantity_gate": quantity_gate,
                "without_quantity_prediction": without_quantity}

    def forward_heads(self, windows, clock):
        """逐样本参考路径；不使用共享前缀，便于逻辑和小样本对照。"""
        windows, lengths = prepare_model_inputs(windows, clock, self.window_bars)
        vectors = [self.encoders[f](windows[f], lengths[f]) for f in FREQUENCIES]
        return self.fuse(vectors, clock)

    def forward(self, windows, clock):
        return self.forward_heads(windows, clock)[:, 0]

    def forward_group(self, windows, latest_clock):
        """股票日共享推理；数量状态完全由历史估计，不接受辅助真值。"""
        return self.forward_group_tasks(windows, latest_clock)["prediction"]

    def forward_group_tasks(self, windows, latest_clock):
        """收益、数量变化及任务路由；标签仅在独立训练目标中使用。"""
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
        tasks = self.fuse_tasks(vectors, clocks)
        return {name: value.reshape(8, batch, -1).transpose(0, 1) for name, value in tasks.items()}
