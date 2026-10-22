"""只对训练计算分块重计算；公共推理路径和权重结构保持不变。"""

import torch
from torch.utils.checkpoint import checkpoint


def grouped_components(model, windows, clock, microbatch, recompute):
    batches = []
    for offset in range(0, len(clock), microbatch):
        part = {frequency: value[offset:offset + microbatch] for frequency, value in windows.items()}
        local_clock = clock[offset:offset + microbatch]
        if recompute:
            result = checkpoint(model.forward_group_components, part, local_clock,
                                use_reentrant=False, preserve_rng_state=True)
        else:
            result = model.forward_group_components(part, local_clock)
        batches.append(result)
    return tuple(torch.cat([result[head] for result in batches], dim=0) for head in range(3))
