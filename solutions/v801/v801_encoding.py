"""复用父模型的单向前缀，提取八个时点的冻结编码和时钟。"""

import torch
import numpy as np
from torch.nn import functional as F

from v801_data import prediction_clock_values
from v801_schema import FREQUENCIES, prepare_model_inputs


def encode_all_clocks(model, windows, latest_clock):
    """每个股票的最长可见输入编码一次，返回八时点表示；只取实际请求键。"""
    if model.training:
        raise ValueError("共享编码仅用于 eval 推理，不支持训练时共享 dropout")
    windows, lengths = prepare_model_inputs(windows, latest_clock, model.window_bars)
    batch = latest_clock.shape[0]
    device = latest_clock.device
    vectors = []
    with torch.autocast(device_type=device.type, enabled=False):
        for frequency in FREQUENCIES:
            encoder = model.encoders[frequency]
            if frequency != "bar1m":
                vector = encoder(windows[frequency], lengths[frequency])
                vectors.append(vector.unsqueeze(0).expand(8, -1, -1).reshape(8 * batch, -1))
                continue
            if encoder.days != 2 or encoder.daily_bars != 240:
                raise ValueError("v801 共享编码要求昨日与今日的固定 1 分钟窗口")
            h = encoder.dropout(F.gelu(encoder.project(windows[frequency].transpose(1, 2))))
            h = h.reshape(batch * 2, 240, -1)
            states, _ = encoder.intraday(h.float())
            states = states.reshape(batch, 2, 240, -1)
            # 单向 GRU 的前缀状态与对各前缀分别补零编码相同；未来状态不参与对应读出。
            last = (torch.arange(8, device=device) * 30 - 1).clamp_min(0)
            current = states[:, 1, last].transpose(0, 1).clone()
            current[0] = 0
            yesterday = states[:, 0, -1].unsqueeze(0).expand(8, -1, -1)
            daily = torch.stack([yesterday, current], dim=2).reshape(8 * batch, 2, -1)
            interday, _ = encoder.interday(encoder.dropout(daily))
            last_day = (torch.arange(8, device=device) > 0).long().repeat_interleave(batch)
            vectors.append(interday[torch.arange(8 * batch, device=device), last_day])
        clocks = torch.eye(8, device=device, dtype=latest_clock.dtype).repeat_interleave(batch, dim=0)
        features = torch.cat([*vectors, clocks], dim=1)
    return features.reshape(8, batch, -1).transpose(0, 1)
