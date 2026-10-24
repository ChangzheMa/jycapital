"""因果门控delta记忆：纯Torch分块计算及独立逐步数学参考。"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def delta_scan(query, key, value, log_decay, write, chunk_size=32, initial_state=None):
    """[B,H,T,D]；键由调用者单位化，返回全部读出和块末状态。

    分块解出每次实际写入的残差，避免除以可能下溢的累计遗忘率。
    实现依据论文的标量门递推，未复制第三方实现或调用专用扩展。
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size必须是正整数')
    if query.ndim != 4 or query.shape != key.shape or value.shape[:3] != key.shape[:3]:
        raise ValueError('query/key/value形状不一致')
    if log_decay.shape != key.shape[:3] or write.shape != log_decay.shape:
        raise ValueError('标量门形状不一致')
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=query.device.type, enabled=False):
        query, key, value, log_decay, write = [x.to(dtype) for x in (query, key, value, log_decay, write)]
        shape = (*key.shape[:2], key.shape[-1], value.shape[-1])
        state = key.new_zeros(shape) if initial_state is None else initial_state.to(dtype)
        if state.shape != shape:
            raise ValueError('初始状态形状不一致')
        outputs = []
        for start in range(0, key.shape[2], chunk_size):
            stop = start + chunk_size
            q, k, v = (x[:, :, start:stop] for x in (query, key, value))
            beta = write[:, :, start:stop]
            cumulative = log_decay[:, :, start:stop].cumsum(-1)
            size = q.shape[2]
            lower = torch.ones(size, size, dtype=torch.bool, device=q.device).tril()
            strict = lower & ~torch.eye(size, dtype=torch.bool, device=q.device)
            difference = cumulative[..., :, None] - cumulative[..., None, :]
            # 上三角可能有很大的正数，必须先遮罩再exp，不能先溢出后乘零。
            transport = torch.exp(difference.masked_fill(~lower, 0)).masked_fill(~lower, 0)
            inherited = cumulative.exp()[..., None] * (k @ state)
            lhs = (beta[..., :, None] * (k @ k.transpose(-1, -2)) * transport).masked_fill(~strict, 0)
            lhs = lhs + torch.eye(size, dtype=dtype, device=q.device)
            residual = torch.linalg.solve_triangular(
                lhs, beta[..., None] * (v - inherited), upper=False, unitriangular=True)
            outputs.append(cumulative.exp()[..., None] * (q @ state)
                           + ((q @ k.transpose(-1, -2)) * transport) @ residual)
            tail = (cumulative[..., -1:] - cumulative).exp()
            state = (cumulative[..., -1, None, None].exp() * state
                     + k.transpose(-1, -2) @ (tail[..., None] * residual))
        output = torch.cat(outputs, dim=2) if outputs else value.new_empty(*value.shape[:2], 0, value.shape[-1])
        return output, state


def delta_recurrent_reference(query, key, value, log_decay, write, initial_state=None):
    """独立逐Bar方程，仅供小型数值与因果检查，不作为正式训练路径。"""
    state = key.new_zeros(*key.shape[:2], key.shape[-1], value.shape[-1]) if initial_state is None else initial_state
    outputs = []
    for t in range(key.shape[2]):
        state = log_decay[..., t, None, None].exp() * state
        k = key[:, :, t]
        previous = (k[..., None, :] @ state).squeeze(-2)
        residual = write[..., t, None] * (value[:, :, t] - previous)
        state = state + k[..., :, None] * residual[..., None, :]
        outputs.append((query[:, :, t, None, :] @ state).squeeze(-2))
    return torch.stack(outputs, dim=2), state


class HiddenRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(value.dtype) * self.weight


class DeltaBlock(nn.Module):
    """日内局部因果混合后，矩阵状态连续贯穿完整交易日历史。"""
    def __init__(self, spec, daily_bars):
        super().__init__()
        width, heads = spec['hidden_dim'], spec['delta_heads']
        self.heads, self.key_dim, self.value_dim = heads, spec['delta_key_dim'], spec['delta_value_dim']
        self.daily_bars, self.chunk_size = daily_bars, spec['delta_chunk_size']
        self.kernel = spec['delta_conv_kernel']
        self.norm = HiddenRMSNorm(width)
        self.query = nn.Linear(width, heads * self.key_dim, bias=False)
        self.key = nn.Linear(width, heads * self.key_dim, bias=False)
        self.value = nn.Linear(width, heads * self.value_dim, bias=False)
        self.convolution = nn.Conv1d(width, width, self.kernel, groups=width, bias=True)
        self.decay = nn.Linear(width, heads)
        self.write = nn.Linear(width, heads)
        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(self.decay.bias, math.log(math.expm1(-math.log(spec['delta_initial_retention']))))
        nn.init.zeros_(self.write.weight)
        nn.init.constant_(self.write.bias, math.log(spec['delta_initial_write'] / (1 - spec['delta_initial_write'])))
        self.output_gate = nn.Linear(width, heads * self.value_dim)
        self.output_norm = HiddenRMSNorm(self.value_dim)
        self.output = nn.Linear(heads * self.value_dim, width)
        self.dropout = nn.Dropout(spec['dropout'])
        self.memory_scale = nn.Parameter(torch.tensor(float(spec['residual_scale_initial'])))
        self.ff_norm = HiddenRMSNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * spec['delta_ff_multiplier']), nn.GELU(),
                                nn.Dropout(spec['dropout']), nn.Linear(width * spec['delta_ff_multiplier'], width))
        self.ff_scale = nn.Parameter(torch.tensor(float(spec['residual_scale_initial'])))

    def forward(self, inputs, available, *, recurrent_reference=False):
        batch, length, width = inputs.shape
        if length % self.daily_bars:
            raise ValueError('日内局部混合要求完整规则日网格')
        hidden = self.norm(inputs).masked_fill(~available[..., None], 0)
        local = hidden.reshape(-1, self.daily_bars, width).transpose(1, 2)
        local = F.silu(self.convolution(F.pad(local, (self.kernel - 1, 0))))
        local = local.transpose(1, 2).reshape(batch, length, width)
        # 不让过去有效Bar通过卷积在整根缺失或未来位置制造一次写入。
        local = local.masked_fill(~available[..., None], 0)
        reshape = lambda x, size: x.reshape(batch, length, self.heads, size).transpose(1, 2)
        q = F.normalize(reshape(self.query(local), self.key_dim).float(), dim=-1, eps=1e-6)
        k = F.normalize(reshape(self.key(local), self.key_dim).float(), dim=-1, eps=1e-6)
        v = reshape(self.value(local), self.value_dim).float()
        decay = -F.softplus(self.decay(hidden).float()).transpose(1, 2)
        write = torch.sigmoid(self.write(hidden).float()).transpose(1, 2)
        decay = decay.masked_fill(~available[:, None, :], 0)
        write = write.masked_fill(~available[:, None, :], 0)
        if recurrent_reference:
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                recalled, _ = delta_recurrent_reference(q, k, v, decay, write)
        else:
            recalled, _ = delta_scan(q, k, v, decay, write, self.chunk_size)
        recalled = self.output_norm(recalled).transpose(1, 2).flatten(-2)
        recalled = recalled * F.silu(self.output_gate(hidden))
        hidden = inputs + self.memory_scale * self.dropout(self.output(recalled))
        hidden = hidden + self.ff_scale * self.dropout(self.ff(self.ff_norm(hidden)))
        return hidden.masked_fill(~available[..., None], 0)
