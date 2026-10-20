"""分层 GRU：先编码日内序列，再积累跨日状态，仅读取已发生位置。"""

import torch
from torch import nn
from torch.nn import functional as F

from v306_schema import (FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs,
                         validate_backbone_config)

DAILY_BARS = {"bar1m": 240, "bar5m": 48, "bar30m": 8}


def validate_model_config(config):
    validate_backbone_config(config, "hierarchical_gru", ("intraday_layers", "interday_layers"))
    expected = {"normalization": "none", "pooling": "last_available_state",
                "recurrent_precision": "float32", "missing_history": "keep_slots_and_flags"}
    if any(config["model"].get(k) != v for k, v in expected.items()):
        raise ValueError("v306 要求 FP32 单向递推、截点读取和保留历史缺失槽位")


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, frequency, length):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.daily_bars = DAILY_BARS[frequency]
        self.days = length // self.daily_bars
        self.project = nn.Linear(INPUT_CHANNELS, hidden)
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


class MultiFrequencyGRU(nn.Module):
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
