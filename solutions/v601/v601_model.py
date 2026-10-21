"""原分辨率的日内因果卷积与跨日 GRU，保留缺失槽位及预测截点。"""

import torch
from torch import nn
from torch.nn import functional as F

from v601_schema import (FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs,
                         validate_backbone_config)

DAILY_BARS = {"bar1m": 240, "bar5m": 48, "bar30m": 8}


def validate_model_config(config):
    validate_backbone_config(config, "hierarchical_gru", ("interday_layers", "kernel_size"))
    spec = config["model"]
    expected = {"normalization": "none", "intraday_encoder": "causal_residual_tcn",
                "pooling": "last_and_visible_mean_then_interday_last",
                "recurrent_precision": "float32", "missing_history": "keep_slots_and_flags"}
    if any(spec.get(k) != v for k, v in expected.items()) or "intraday_layers" in spec:
        raise ValueError("v601 要求日内因果卷积及 FP32 跨日递推")
    dilations = spec.get("intraday_dilations", {})
    if set(dilations) != set(FREQUENCIES):
        raise ValueError("必须显式配置各频率的日内膨胀率")
    for values in dilations.values():
        if not isinstance(values, list) or not values or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError("日内膨胀率必须为非空正整数列表")


class CausalResidual(nn.Module):
    def __init__(self, hidden, kernel_size, dilation, dropout):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(F.gelu(self.conv1(F.pad(x, (self.left_padding, 0)))))
        h = self.dropout(F.gelu(self.conv2(F.pad(h, (self.left_padding, 0)))))
        return x + h


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, frequency, length):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.daily_bars = DAILY_BARS[frequency]
        self.days = length // self.daily_bars
        self.project = nn.Linear(INPUT_CHANNELS, hidden)
        self.dropout = nn.Dropout(spec["dropout"])
        self.intraday = nn.Sequential(*[
            CausalResidual(hidden, spec["kernel_size"], d, spec["dropout"])
            for d in spec["intraday_dilations"][frequency]])
        self.daily_readout = nn.Linear(2 * hidden, hidden)
        self.interday = nn.GRU(hidden, hidden, num_layers=spec["interday_layers"], batch_first=True,
                              dropout=spec["dropout"] if spec["interday_layers"] > 1 else 0)

    def forward(self, x, lengths):
        batch = x.shape[0]
        h = self.dropout(F.gelu(self.project(x.transpose(1, 2))))
        h = h.reshape(batch * self.days, self.daily_bars, -1).transpose(1, 2)
        states = self.intraday(h)
        day_lengths = (lengths[:, None] - torch.arange(
            self.days, device=x.device)[None, :] * self.daily_bars).clamp(0, self.daily_bars)
        count = day_lengths.flatten()
        last = states[torch.arange(batch * self.days, device=x.device), :, (count - 1).clamp_min(0)]
        visible = torch.arange(self.daily_bars, device=x.device)[None, :] < count[:, None]
        # 只池化预测截点前的位置；历史缺失位置仍属于固定网格，由输入掩码表达。
        mean = states.float().masked_fill(~visible[:, None, :], 0).sum(dim=2) / count.clamp_min(1)[:, None]
        daily = F.gelu(self.daily_readout(torch.cat([last.float(), mean], dim=1)))
        daily = daily.reshape(batch, self.days, -1).masked_fill((day_lengths == 0)[:, :, None], 0)
        with torch.autocast(device_type=x.device.type, enabled=False):
            interday, _ = self.interday(self.dropout(daily.float()))
            last_day = (lengths - 1) // self.daily_bars
            return interday[torch.arange(batch, device=x.device), last_day]


class MultiFrequencyGRU(nn.Module):
    """保留分层 GRU 家族接口；日内模块由并行因果卷积实现。"""

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
