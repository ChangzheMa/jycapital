"""v1102 日内多时点共享编码推理，输出收益主头。"""

import torch
import numpy as np

from v1102_data import prediction_clock_values


def predict_all_clocks(model, windows, latest_clock):
    if model.training:
        raise ValueError("shared inference requires eval mode")
    return model.forward_group(windows, latest_clock)[..., 0]


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
