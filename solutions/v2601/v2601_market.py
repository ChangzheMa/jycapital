"""完整当前市场条件，在新增原始分支的非线性提取前调制；不接收标签。"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from v2601_schema import FREQUENCIES, TRADING_DAY_WINDOWS

MARKET_KEYS = ('market_snapshots', 'market_membership', 'market_target_rows')


def extract_snapshots(windows, snapshot_bars, window_bars, indices=None):
    if set(windows) != set(FREQUENCIES):
        raise ValueError('原始快照读取只接受三频行情')
    result = []
    batch = next(iter(windows.values())).shape[0]
    for f in FREQUENCIES:
        values = windows[f]
        if values.shape != (batch, 75, window_bars[f]):
            raise ValueError('原始快照窗口形状不符')
        if indices is None:
            bars = 240 // int(f[3:-1])
            ends = TRADING_DAY_WINDOWS[f]['previous_days'] * bars + torch.arange(8, device=values.device) * (bars // 8)
            positions = ends[:, None] - snapshot_bars[f] + torch.arange(snapshot_bars[f], device=values.device)[None]
        else:
            positions = indices[f]
        result.append(values[:, :, positions].permute(0, 2, 3, 1).reshape(batch, 8, -1))
    return torch.cat(result, dim=-1).contiguous()


def validate_market_config(config):
    spec = config['model'].get('market_conditioning', {})
    for name in ('context_dim', 'context_tokens', 'attention_heads', 'raw_hidden_dim'):
        if type(spec.get(name)) is not int or spec[name] < 1:
            raise ValueError('市场条件维度须为正整数: ' + name)
    if spec['context_dim'] % spec['attention_heads']:
        raise ValueError('市场条件宽度必须能被头数整除')
    if set(spec.get('snapshot_bars', {})) != set(FREQUENCIES):
        raise ValueError('近期原始观测必须包含三频')
    for frequency, size in spec['snapshot_bars'].items():
        earliest = TRADING_DAY_WINDOWS[frequency]['previous_days'] * (240 // int(frequency[3:-1]))
        if type(size) is not int or not 0 < size <= earliest:
            raise ValueError('近期观测数量超出已固定窗口')
    expected = {'context_activation': 'tanh', 'modulation': 'affine_before_activation',
                'modulation_initial': 'identity', 'correction_initial': 'zero',
                'membership': 'current_clock_full_universe'}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError('原始市场条件契约不一致')
    scale = spec.get('residual_scale_initial')
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('市场分支初始残差尺度须在(0,1]内')


class RawMarketConditioning(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_market_config(config)
        spec = config['model']['market_conditioning']
        self.width = spec['context_dim']
        self.heads = spec['attention_heads']
        self.head_width = self.width // self.heads
        self.features = sum(spec['snapshot_bars'].values()) * 75
        self.window_bars = dict(config['data']['window_bars'])
        self.snapshot_bars = dict(spec['snapshot_bars'])
        raw_width, drop = spec['raw_hidden_dim'], config['model']['dropout']
        self.context_encoder = nn.Sequential(nn.Linear(self.features + 8, self.width), nn.GELU(),
            nn.Dropout(drop), nn.Linear(self.width, self.width), nn.Tanh())
        self.pool_scores = nn.Linear(self.width, spec['context_tokens'], bias=False)
        self.key = nn.Linear(self.width, self.width, bias=False)
        self.value = nn.Linear(self.width, self.width)
        self.query = nn.Linear(self.width, self.width)
        self.raw_projection = nn.Linear(self.features + 8, raw_width)
        self.conditioner = nn.Sequential(nn.Linear(2 * self.width, raw_width), nn.GELU(),
                                         nn.Linear(raw_width, 2 * raw_width))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        self.readout = nn.Sequential(nn.Dropout(drop), nn.Linear(raw_width, raw_width), nn.GELU(),
                                     nn.Dropout(drop), nn.Linear(raw_width, 2))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(spec['residual_scale_initial'])))
        self.register_buffer('clock_eye', torch.eye(8), persistent=False)
        for f in FREQUENCIES:
            bars = 240 // int(f[3:-1])
            ends = TRADING_DAY_WINDOWS[f]['previous_days'] * bars + torch.arange(8) * (bars // 8)
            indices = ends[:, None] - self.snapshot_bars[f] + torch.arange(self.snapshot_bars[f])[None, :]
            self.register_buffer(f + '_snapshot_indices', indices, persistent=False)

    def snapshot_features(self, windows):
        return extract_snapshots(windows, self.snapshot_bars, self.window_bars,
                                 {f: getattr(self, f + '_snapshot_indices') for f in FREQUENCIES})

    def with_clock(self, values):
        return torch.cat((values, self.clock_eye.to(values.dtype)[None].expand(len(values), -1, -1)), dim=-1)

    def prepare_context(self, snapshots, membership):
        if snapshots.ndim != 3 or snapshots.shape[1:] != (8, self.features) or len(snapshots) < 1:
            raise ValueError('市场上下文要求非空的股票×八时点×原始观测')
        if membership.shape != snapshots.shape[:2] or membership.dtype != torch.bool or membership.device != snapshots.device:
            raise ValueError('市场上下文必须有逐时点布尔成员标志')
        encoded = self.context_encoder(self.with_clock(snapshots))
        logits = self.pool_scores(encoded).permute(1, 2, 0).float()
        visible = membership.T[:, None, :]
        logits = logits.masked_fill(~visible, float('-inf'))
        logits = torch.where(visible.any(dim=-1, keepdim=True), logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1) * visible
        with torch.autocast(snapshots.device.type, enabled=False):
            tokens = torch.einsum('cmn,ncd->cmd', weights, encoded.float())
        return {'snapshots': snapshots, 'membership': membership, 'encoded': encoded,
                'keys': self.key(tokens), 'values': self.value(tokens)}

    def correction(self, context, rows):
        if rows.ndim != 1 or rows.dtype != torch.int64:
            raise ValueError('目标股票必须显式映射到完整上下文')
        if rows.numel() and ((rows < 0).any() or (rows >= len(context['snapshots'])).any()):
            raise ValueError('目标股票上下文索引越界')
        own = context['encoded'].index_select(0, rows)
        raw = context['snapshots'].index_select(0, rows)
        present = context['membership'].index_select(0, rows)
        query = self.query(own).reshape(len(rows), 8, self.heads, self.head_width)
        keys = context['keys'].reshape(8, -1, self.heads, self.head_width)
        values = context['values'].reshape(8, -1, self.heads, self.head_width)
        with torch.autocast(raw.device.type, enabled=False):
            score = torch.einsum('bchd,cmhd->bchm', query.float(), keys.float()) / math.sqrt(self.head_width)
            attention = torch.softmax(score, dim=-1)
            market = torch.einsum('bchm,cmhd->bchd', attention, values.float()).flatten(-2)
        gamma, beta = self.conditioner(torch.cat((own, market), dim=-1)).chunk(2, dim=-1)
        features = F.gelu((1 + gamma) * self.raw_projection(self.with_clock(raw)) + beta)
        correction = self.residual_scale * self.readout(features).float()
        correction = torch.where(present[..., None], correction, torch.zeros_like(correction))
        count = present.sum().clamp_min(1) * gamma.shape[-1]
        details = {'correction': correction, 'membership': present,
                   'modulation_scale_rms': ((gamma.detach().float().square() * present[..., None]).sum() / count).sqrt(),
                   'modulation_shift_rms': ((beta.detach().float().square() * present[..., None]).sum() / count).sqrt()}
        return correction, details
