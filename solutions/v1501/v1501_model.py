"""完整市场时空 Transformer：因果原始序列编码、股票交互与日内因果状态。"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from v1501_schema import FREQUENCIES, INPUT_CHANNELS, prepare_model_inputs, validate_backbone_config

DAILY_BARS = {'bar1m':240, 'bar5m':48, 'bar30m':8}


def validate_model_config(config):
    validate_backbone_config(config, 'patch_transformer',
        ('attention_heads','temporal_layers','market_cycles','ff_multiplier','output_heads','encoding_stock_batch_size'))
    spec = config['model']
    expected = {'normalization':'pre_layer_norm','clock_readout':'last_closed_patch',
                'market_membership':'observed_universe_at_each_clock','stock_identity_embedding':False,
                'temporal_attention':'strictly_causal','activation_checkpoint':True,'output_heads':2}
    if any(spec.get(key) != value for key,value in expected.items()):
        raise ValueError('v1501 必须采用已登记的完整市场因果结构')
    if any(spec[key] % spec['attention_heads'] for key in ('hidden_dim','fusion_hidden_dim')):
        raise ValueError('注意力宽度必须整除头数')
    for frequency,bars in DAILY_BARS.items():
        patch = spec.get('patch_sizes',{}).get(frequency)
        if type(patch) is not int or patch < 1 or (bars//8) % patch:
            raise ValueError('Patch 必须整除每个预测时点新增的 Bar，不能跨越截点')
    scale = spec.get('residual_scale_initial')
    if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('残差尺度须在 (0,1] 内')


class AttentionBlock(nn.Module):
    def __init__(self, width, spec):
        super().__init__()
        self.heads, self.head_width = spec['attention_heads'], width//spec['attention_heads']
        self.norm_attention, self.norm_ff = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.projection = nn.Linear(width,3*width), nn.Linear(width,width)
        self.ff = nn.Sequential(nn.Linear(width,width*spec['ff_multiplier']),nn.GELU(),nn.Dropout(spec['dropout']),
                                nn.Linear(width*spec['ff_multiplier'],width))
        self.dropout = nn.Dropout(spec['dropout'])
        self.attention_scale = nn.Parameter(torch.tensor(float(spec['residual_scale_initial'])))
        self.ff_scale = nn.Parameter(torch.tensor(float(spec['residual_scale_initial'])))

    def forward(self, value, present=None, causal=False):
        batch, length, width = value.shape
        q,k,v = self.qkv(self.norm_attention(value)).reshape(batch,length,3,self.heads,self.head_width).permute(2,0,3,1,4)
        logits = (q @ k.transpose(-1,-2)).float()/math.sqrt(self.head_width)
        allowed = torch.ones((batch,1,length,length),dtype=torch.bool,device=value.device)
        if present is not None:
            allowed = allowed & present[:,None,None,:]
        if causal:
            allowed = allowed & torch.ones((length,length),dtype=torch.bool,device=value.device).tril()
        weights = torch.softmax(logits.masked_fill(~allowed,torch.finfo(logits.dtype).min),dim=-1)
        weights = torch.where(allowed,weights,0)
        context = (self.dropout(weights.to(v.dtype)) @ v).transpose(1,2).reshape(batch,length,width)
        value = value + self.attention_scale*self.dropout(self.projection(context))
        value = value + self.ff_scale*self.dropout(self.ff(self.norm_ff(value)))
        return torch.where(present[...,None],value,0) if present is not None else value


class CausalPatchEncoder(nn.Module):
    def __init__(self, frequency, length, spec):
        super().__init__()
        self.patch = spec['patch_sizes'][frequency]
        self.tokens = length//self.patch
        self.bars = DAILY_BARS[frequency]
        self.previous_tokens = (length-self.bars)//self.patch
        self.project = nn.Linear(INPUT_CHANNELS*self.patch,spec['hidden_dim'])
        self.position = nn.Parameter(torch.empty(1,self.tokens,spec['hidden_dim']))
        nn.init.normal_(self.position,std=.02)
        self.dropout = nn.Dropout(spec['dropout'])
        self.layers = nn.ModuleList([AttentionBlock(spec['hidden_dim'],spec) for _ in range(spec['temporal_layers'])])

    def forward(self, window):
        value = window.transpose(1,2).reshape(len(window),self.tokens,INPUT_CHANNELS*self.patch)
        value = self.dropout(F.gelu(self.project(value)) + self.position)
        for layer in self.layers:
            value = layer(value,causal=True)
        cutoffs = self.previous_tokens + torch.arange(8,device=value.device)*(self.bars//8//self.patch)-1
        return value[:,cutoffs]


class MarketClockBlock(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.market = AttentionBlock(spec['fusion_hidden_dim'],spec)
        self.clock = AttentionBlock(spec['fusion_hidden_dim'],spec)

    def forward(self, value, membership):
        moments = []
        for clock in range(8):
            indices = torch.nonzero(membership[:,clock],as_tuple=True)[0]
            current = torch.zeros_like(value[:,clock])
            if len(indices):
                # 每个时点只对当前股票池做完整交互，未来才出现的股票不参与。
                attended = self.market(value[indices,clock][None])[0]
                current = current.index_copy(0,indices,attended)
            moments.append(current)
        value = torch.stack(moments,dim=1)
        return self.clock(value,present=membership,causal=True)


class MultiFrequencyPatchTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_model_config(config)
        spec = config['model']
        self.window_bars = dict(config['data']['window_bars'])
        self.encoding_stock_batch_size = spec['encoding_stock_batch_size']
        self.activation_checkpoint = spec['activation_checkpoint']
        self.encoders = nn.ModuleDict({frequency:CausalPatchEncoder(frequency,self.window_bars[frequency],spec)
                                       for frequency in FREQUENCIES})
        self.fusion = nn.Sequential(nn.Linear(3*spec['hidden_dim']+8,spec['fusion_hidden_dim']),
                                    nn.GELU(),nn.Dropout(spec['dropout']))
        self.market_clock = nn.ModuleList([MarketClockBlock(spec) for _ in range(spec['market_cycles'])])
        self.readout = nn.Linear(spec['fusion_hidden_dim'],2)

    def encode_group(self, windows, latest_clock):
        windows,_ = prepare_model_inputs(windows,latest_clock,self.window_bars)
        representations = [self.encoders[frequency](windows[frequency]) for frequency in FREQUENCIES]
        clocks = torch.eye(8,device=latest_clock.device,dtype=representations[0].dtype)[None].expand(len(latest_clock),-1,-1)
        return self.fusion(torch.cat([*representations,clocks],dim=-1))

    def decode_group(self, representations, membership):
        if membership.shape != representations.shape[:2] or membership.dtype != torch.bool:
            raise ValueError('完整市场成员掩码须为 [股票日,8] 布尔值')
        value = torch.where(membership[...,None],representations,0)
        for layer in self.market_clock:
            value = layer(value,membership)
        return torch.where(membership[...,None],self.readout(value).float(),0)

    def forward_group(self, windows, latest_clock, membership=None):
        available = torch.arange(8,device=latest_clock.device)[None] <= latest_clock.argmax(dim=-1)[:,None]
        membership = available if membership is None else membership
        if membership.shape != available.shape or membership.dtype != torch.bool or (membership & ~available).any():
            raise ValueError('成员掩码不能包含代表行尚不可见的时点')
        representations = []
        for start in range(0,len(latest_clock),self.encoding_stock_batch_size):
            stop = start+self.encoding_stock_batch_size
            part = {frequency:window[start:stop] for frequency,window in windows.items()}
            clock = latest_clock[start:stop]
            if self.training and torch.is_grad_enabled() and self.activation_checkpoint:
                value = checkpoint(self.encode_group,part,clock,use_reentrant=False,preserve_rng_state=True)
            else:
                value = self.encode_group(part,clock)
            representations.append(value)
        return self.decode_group(torch.cat(representations,dim=0),membership)
