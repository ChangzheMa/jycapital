"""股票日分批编码，然后按完整时点进行一次集合预测。"""

import torch
import numpy as np

from v802_data import prediction_clock_values


def predict_day(model, block, device, *, stock_batch_size=128):
    if model.training:
        raise ValueError("平台预测须使用 eval 模式")
    if isinstance(stock_batch_size, bool) or not isinstance(stock_batch_size, int) or stock_batch_size < 1:
        raise ValueError("股票日编码批量必须为正整数")
    if not len(block.keys):
        return np.empty(0, np.float64)
    if len(np.unique(block.day_indices)) != 1:
        raise ValueError("共享编码入口一次只接收一个交易日")
    _, group = np.unique(block.stock_indices, return_inverse=True)
    clocks = prediction_clock_values(block.times).argmax(axis=1)
    order = np.lexsort((clocks, group))
    ends = np.r_[np.flatnonzero(np.diff(group[order])) + 1, len(order)] - 1
    representatives = order[ends]
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(representatives), stock_batch_size):
            rows = representatives[start:start + stock_batch_size]
            windows, latest = block.batch(rows)
            windows = {f: torch.from_numpy(value).to(device) for f, value in windows.items()}
            embeddings.append(model.encode_shared(windows, torch.from_numpy(latest).to(device)))
        embeddings = torch.cat(embeddings, dim=0)
        output = np.empty(len(block.keys), np.float64)
        for clock_id in np.unique(clocks):
            rows = np.flatnonzero(clocks == clock_id)
            stocks = torch.as_tensor(group[rows], device=device)
            scores = model.forward_features(embeddings[stocks, int(clock_id)])
            output[rows] = (scores / block.config["inference"]["factor_scale"]).cpu().numpy()
    return output
