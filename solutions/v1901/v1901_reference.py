"""逐时点重新编码完整股票池的独立参考推理。"""
import torch
import numpy as np


def predict_day_reference(model, block, device, *, stock_batch_size=128):
    if model.training:
        raise ValueError("参考推理要求 eval 模式")
    output = np.full(len(block.keys), np.nan, np.float64)
    with torch.inference_mode():
        for _, rows in block.keys.groupby("date", sort=True).groups.items():
            rows = np.asarray(rows, dtype=np.int64)
            parts, clocks = [], []
            for start in range(0, len(rows), stock_batch_size):
                windows, clock = block.batch(rows[start:start + stock_batch_size])
                windows = {key: torch.from_numpy(value).to(device) for key, value in windows.items()}
                clock = torch.from_numpy(clock).to(device)
                parts.append(model.encode_reference(windows, clock))
                clocks.append(clock)
            encoded = torch.cat(parts, dim=0)
            clock = torch.cat(clocks, dim=0)
            if not torch.all(clock == clock[0]):
                raise ValueError("参考截面含多个预测时点")
            index = int(clock[0].argmax())
            states = encoded.new_zeros(len(encoded), 8, encoded.shape[-1])
            states[:, index] = encoded
            membership = torch.zeros(len(encoded), 8, dtype=torch.bool, device=device)
            membership[:, index] = True
            output[rows] = model.decode_group(states, membership)[:, index, 0].cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("参考推理未覆盖全部请求键")
    return output
