"""完整市场 fast：共享股票日 GRU 编码，潜在因子先验在完整截面计算。"""

import numpy as np
from v1901_market import predict_complete_days


def predict_day(model, block, device, *, stock_batch_size=1200):
    if not len(block.keys):
        return np.empty(0,np.float64)
    if len(np.unique(block.day_indices))!=1:
        raise ValueError('fast 入口每次接收完整交易日上下文')
    return predict_complete_days(model,block,device,block.config)
