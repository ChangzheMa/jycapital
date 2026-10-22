"""保持原始优化路径不变，用独立 FP32 权重副本维护 EMA。"""

import copy
import math

import torch


class WeightEMA:
    def __init__(self, model, decay, warmup_epochs):
        if not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError("EMA decay 必须是 (0, 1) 内的有限数")
        if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError("EMA warmup_epochs 必须是非负整数")
        self.decay, self.warmup_epochs = decay, warmup_epochs
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        source = dict(model.named_parameters())
        shadow = dict(self.model.named_parameters())
        if source.keys() != shadow.keys() or any(p.dtype != torch.float32 for p in source.values()):
            raise ValueError("EMA 要求对应的 FP32 参数")
        self.parameters = [(shadow[name], value) for name, value in source.items()]
        source_buffers, shadow_buffers = dict(model.named_buffers()), dict(self.model.named_buffers())
        if source_buffers.keys() != shadow_buffers.keys():
            raise ValueError("EMA 缓冲区不对应")
        self.buffers = [(shadow_buffers[name], value) for name, value in source_buffers.items()]
        self.updates = 0

    @torch.no_grad()
    def update(self, epoch):
        for shadow, source in self.parameters:
            if epoch <= self.warmup_epochs:
                shadow.copy_(source)
            else:
                shadow.lerp_(source, 1 - self.decay)
        # 当前 GRU 没有统计缓冲区；若存在非参数状态，保持原值而不做数值平均。
        for shadow, source in self.buffers:
            shadow.copy_(source)
        self.updates += 1

    def restore(self, weights, updates, optimizer_steps):
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0 or updates != optimizer_steps:
            raise ValueError("EMA 更新次数与已完成优化步数不一致")
        self.model.load_state_dict(weights, strict=True)
        self.updates = updates
