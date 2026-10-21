"""冻结 v501 的日内多时点推理复用，不改变模型参数或训练路径。"""

import torch
import numpy as np
from torch.nn import functional as F

from v501_data import prediction_clock_values
from v501_schema import FREQUENCIES, prepare_model_inputs


def predict_all_clocks(model, windows, latest_clock):
    """每个股票的最长可见输入编码一次，返回八时点分数；调用者只取请求键。"""
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
                raise ValueError("v501 共享编码要求昨日与今日的固定 1 分钟窗口")
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
        scores = model.head(torch.cat([*vectors, clocks], dim=1)).squeeze(1)
    return scores.reshape(8, batch).transpose(0, 1)


def predict_day(model, block, device, *, stock_batch_size=128):
    """按股票日复用取窗，保留动态股票池、缺失时点及原键顺序。"""
    if isinstance(stock_batch_size, bool) or not isinstance(stock_batch_size, int) or stock_batch_size < 1:
        raise ValueError("股票日批量必须为正整数")
    if model.training:
        raise ValueError("共享编码仅用于 eval 推理")
    if not len(block.keys):
        return np.empty(0, np.float64)
    if len(np.unique(block.day_indices)) != 1:
        raise ValueError("共享编码入口每次只接收一个交易日")
    _, group = np.unique(block.stock_indices, return_inverse=True)
    count = int(group.max()) + 1
    clock_ids = prediction_clock_values(block.times).argmax(axis=1)
    # 不依赖 keys 原本排序；每只股票选择请求内最晚时点，未来仍由原 batch 路径清零。
    order = np.lexsort((clock_ids, group))
    ends = np.r_[np.flatnonzero(np.diff(group[order])) + 1, len(order)] - 1
    representatives = order[ends]
    values = np.empty((count, 8), np.float64)
    with torch.inference_mode():
        for offset in range(0, count, stock_batch_size):
            selected = representatives[offset:offset + stock_batch_size]
            windows, latest = block.batch(selected)
            tensors = {f: torch.from_numpy(x).to(device) for f, x in windows.items()}
            scores = predict_all_clocks(model, tensors, torch.from_numpy(latest).to(device))
            values[offset:offset + len(selected)] = (
                scores / block.config["inference"]["factor_scale"]).cpu().numpy()
    return values[group, clock_ids]
