"""v2601 日内多时点共享编码推理，输出收益主头。"""

import torch
import numpy as np

from v2601_data import prediction_clock_values
from v2601_context import index_view, predict_groups, native_batches, collect_context
from v2601_market import MARKET_KEYS


def predict_all_clocks(model, windows, latest_clock, market_state=None, target_rows=None):
    if model.training:
        raise ValueError("shared inference requires eval mode")
    return model.forward_group(windows, latest_clock, market_state, target_rows)[..., 0]


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
    view = index_view(block)
    config = dict(block.config, inference=dict(block.config['inference'], stock_day_batch_size=stock_batch_size))
    scores = predict_groups(model, view, device, config, native_batches)
    result = np.empty(len(block.keys), np.float64)
    view.index.assign_predictions(result, np.arange(len(view.index.samples)), scores)
    return result


def predict_day_reference(model, block, device, *, sample_batch_size):
    """逐样本个股编码，每批重新计算完整市场表示，作为独立参考路径。"""
    if model.training or len(np.unique(block.day_indices)) != 1:
        raise ValueError('参考入口要求单日和eval模式')
    view = index_view(block)
    groups = np.arange(len(view.index.samples))
    snapshots, membership = collect_context(model, view, groups, native_batches,
        block.config['training']['market_conditioning']['context_stock_day_batch_size'], device)
    mapping = np.full(len(block.keys), -1, np.int64)
    for clock in range(8):
        rows = view.index.samples[:, clock]
        present = rows >= 0
        mapping[rows[present]] = groups[present]
    if (mapping < 0).any():
        raise ValueError('参考市场股票映射不完整')
    result = np.empty(len(block.keys), np.float64)
    with torch.inference_mode():
        for offset in range(0, len(block.keys), sample_batch_size):
            selected = np.arange(offset, min(offset + sample_batch_size, len(block.keys)))
            raw, clock = block.batch(selected)
            windows = {f: torch.from_numpy(x).to(device) for f, x in raw.items()}
            windows.update({MARKET_KEYS[0]: snapshots, MARKET_KEYS[1]: membership,
                            MARKET_KEYS[2]: torch.as_tensor(mapping[selected], device=device)})
            values = model(windows, torch.from_numpy(clock).to(device))
            result[selected] = (values / block.config['inference']['factor_scale']).cpu().numpy()
    return result
