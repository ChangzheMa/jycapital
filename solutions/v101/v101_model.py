"""多频 Patch Transformer：联合字段分块、历史双向注意力与 CLS 汇总。"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from v101_schema import (FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs,
                         validate_backbone_config)


def validate_model_config(config):
    validate_backbone_config(config, "patch_transformer", ("layers", "attention_heads", "feedforward_dim"))
    spec = config["model"]
    expected = {"normalization": "pre_layernorm", "pooling": "cls",
                "attention_backend": "math", "padding_mask": "future_patches_only"}
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("v101 要求 Pre-LN、CLS 汇总、math 注意力和未来 patch 掩码")
    if spec["hidden_dim"] % spec["attention_heads"]:
        raise ValueError("隐藏宽度必须能被注意力头数整除")
    patches = spec.get("patch_bars", {})
    if set(patches) != set(FREQUENCIES):
        raise ValueError("每个频率必须配置 patch 长度")
    for frequency, size in patches.items():
        if isinstance(size, bool) or not isinstance(size, int) or size < 1 or \
                config["data"]["window_bars"][frequency] % size:
            raise ValueError("patch 长度必须为整除窗口的正整数")


class AttentionBlock(nn.Module):
    def __init__(self, spec):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.heads = spec["attention_heads"]
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv, self.out = nn.Linear(hidden, hidden * 3), nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(spec["dropout"])
        self.ffn = nn.Sequential(nn.Linear(hidden, spec["feedforward_dim"]), nn.GELU(),
                                 nn.Dropout(spec["dropout"]), nn.Linear(spec["feedforward_dim"], hidden))

    def forward(self, x, available):
        batch, length, hidden = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, hidden // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # 明确选用 math 后端，使确定性 CUDA 训练不依赖 FlashAttention 反向路径。
        with sdpa_kernel(SDPBackend.MATH):
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=available[:, None, None, :],
                dropout_p=self.dropout.p if self.training else 0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        x = x + self.dropout(self.out(attended))
        return x + self.dropout(self.ffn(self.norm2(x)))


class FrequencyEncoder(nn.Module):
    def __init__(self, spec, frequency, window_bars):
        super().__init__()
        hidden = spec["hidden_dim"]
        self.patch = spec["patch_bars"][frequency]
        self.tokens = window_bars // self.patch
        self.project = nn.Linear(INPUT_CHANNELS * self.patch, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.position = nn.Parameter(torch.empty(1, self.tokens + 1, hidden))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([AttentionBlock(spec) for _ in range(spec["layers"])])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, lengths):
        batch = x.shape[0]
        patches = x.transpose(1, 2).reshape(batch, self.tokens, INPUT_CHANNELS * self.patch)
        h = torch.cat([self.cls.expand(batch, -1, -1), self.project(patches)], dim=1) + self.position
        # 部分可用 patch 保留已知 Bar；全未来 patch 不作为 key/value。
        available = torch.arange(self.tokens, device=x.device)[None, :] * self.patch < lengths[:, None]
        available = torch.cat([torch.ones(batch, 1, dtype=torch.bool, device=x.device), available], dim=1)
        for block in self.blocks:
            h = block(h, available)
        return self.norm(h[:, 0])


class MultiFrequencyPatchTransformer(nn.Module):
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
