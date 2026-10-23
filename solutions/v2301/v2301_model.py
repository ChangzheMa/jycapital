"""近期原始Bar与历史收盘槽位直接进入共享参数MLP集成。"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from v2301_schema import FREQUENCIES, INPUT_CHANNELS, TRADING_DAY_WINDOWS, prepare_model_inputs, validate_input_config


def validate_model_config(config):
    validate_input_config(config)
    spec=config['model']
    expected={'family':'snapshot_mlp','activation':'relu','normalization':'none','output_heads':2,
              'daily_anchor_frequency':'bar30m','daily_anchor_position':'last_bar_of_completed_day',
              'member_parameterization':'rank_one_shared_linear',
              'member_initialization':'first_input_rademacher_other_scales_one',
              'prediction_aggregation':'arithmetic_mean',
              'feature_order':'frequency_recent_oldest_to_newest_then_daily_anchors_then_clock'}
    if any(spec.get(key)!=value for key,value in expected.items()):raise ValueError('原始状态MLP配置不匹配')
    for key in ('ensemble_size','width','depth','daily_anchor_days'):
        if isinstance(spec.get(key),bool) or not isinstance(spec.get(key),int) or spec[key]<1:
            raise ValueError('模型规模须为正整数: '+key)
    if spec['daily_anchor_days']>TRADING_DAY_WINDOWS['bar30m']['previous_days']:
        raise ValueError('历史收盘槽位不能超出原输入窗口')
    if set(spec.get('snapshot_bars',{}))!=set(FREQUENCIES):raise ValueError('必须声明三个频率的近期Bar数')
    for frequency,count in spec['snapshot_bars'].items():
        limit=TRADING_DAY_WINDOWS[frequency]['previous_days']*(240//int(frequency[3:-1]))
        if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=limit:
            raise ValueError('近期Bar数不能超出最早预测时点的历史窗口')
    dropout=spec.get('dropout')
    if isinstance(dropout,bool) or not isinstance(dropout,(int,float)) or not math.isfinite(dropout) or not 0<=dropout<1:
        raise ValueError('Dropout须在[0,1)')


class EnsembleLinear(nn.Module):
    """共享W与成员专属r/s/b，不构造每个成员的完整权重矩阵。"""
    def __init__(self,input_dim,output_dim,members,*,first=False):
        super().__init__()
        self.linear=nn.Linear(input_dim,output_dim,bias=False)
        self.r=nn.Parameter(torch.ones(members,input_dim))
        self.s=nn.Parameter(torch.ones(members,output_dim))
        initial=torch.empty(output_dim).uniform_(-1/math.sqrt(input_dim),1/math.sqrt(input_dim))
        self.bias=nn.Parameter(initial.repeat(members,1))
        if first:
            with torch.no_grad():self.r.copy_(torch.randint(0,2,self.r.shape)*2-1)

    def affine_output(self,value):
        return value*self.s.to(value.dtype)+self.bias.to(value.dtype)

    def forward(self,x):
        return self.affine_output(self.linear(x*self.r.to(x.dtype)))

    def forward_group_parts(self,recent,anchors,clocks):
        """同一股票日的历史部分、相同预测时钟的常数部分只各投影一次。"""
        recent_width,anchor_width=recent.shape[-1],anchors.shape[-1]
        split=recent_width+anchor_width
        weight=self.linear.weight
        r=self.r.to(recent.dtype)
        local=F.linear(recent.unsqueeze(-2)*r[:,:recent_width],weight[:,:recent_width])
        past=F.linear(anchors.unsqueeze(-2)*r[:,recent_width:split],weight[:,recent_width:split])
        clock=F.linear(clocks.unsqueeze(-2)*r[:,split:],weight[:,split:])
        return self.affine_output(local+past[:,None]+clock[None])


class MultiFrequencySnapshotMLP(nn.Module):
    def __init__(self,config):
        super().__init__();validate_model_config(config)
        spec=config['model'];self.window_bars=dict(config['data']['window_bars'])
        self.members=spec['ensemble_size'];self.snapshot_bars=dict(spec['snapshot_bars'])
        self.anchor_days=spec['daily_anchor_days']
        self.recent_dim=INPUT_CHANNELS*sum(self.snapshot_bars.values())
        self.anchor_dim=INPUT_CHANNELS*self.anchor_days
        self.input_dim=self.recent_dim+self.anchor_dim+8
        for frequency in FREQUENCIES:
            bars=240//int(frequency[3:-1]);previous=TRADING_DAY_WINDOWS[frequency]['previous_days']
            ends=previous*bars+torch.arange(8)*(bars//8)
            indices=ends[:,None]-torch.arange(self.snapshot_bars[frequency],0,-1)[None,:]
            self.register_buffer('recent_'+frequency,indices,persistent=False)
        prior=TRADING_DAY_WINDOWS['bar30m']['previous_days']
        self.register_buffer('anchor_indices',torch.arange(prior-self.anchor_days,prior)*8+7,persistent=False)
        widths=[self.input_dim]+[spec['width']]*spec['depth']
        self.layers=nn.ModuleList([EnsembleLinear(widths[i],widths[i+1],self.members,first=i==0)
                                  for i in range(spec['depth'])])
        self.dropout=nn.Dropout(spec['dropout'])
        self.head_weight=nn.Parameter(torch.empty(self.members,spec['width'],2))
        self.head_bias=nn.Parameter(torch.zeros(self.members,2))
        nn.init.uniform_(self.head_weight,-1/math.sqrt(spec['width']),1/math.sqrt(spec['width']))

    def group_features(self,windows,latest_clock):
        windows,_=prepare_model_inputs(windows,latest_clock,self.window_bars)
        pieces=[]
        for frequency in FREQUENCIES:
            x=windows[frequency].transpose(1,2)
            pieces.append(x[:,getattr(self,'recent_'+frequency)].flatten(-2))
        recent=torch.cat(pieces,dim=-1)
        anchors=windows['bar30m'].transpose(1,2)[:,self.anchor_indices].flatten(1)
        clocks=torch.eye(8,device=latest_clock.device,dtype=recent.dtype)
        return recent,anchors,clocks

    def sample_features(self,windows,clock):
        """单时点独立取值与完整向量投影，供参考推理和针对性检查。"""
        windows,_=prepare_model_inputs(windows,clock,self.window_bars)
        ids=clock.argmax(dim=1);batch=torch.arange(len(clock),device=clock.device)
        pieces=[]
        for frequency in FREQUENCIES:
            indices=getattr(self,'recent_'+frequency)[ids]
            x=windows[frequency].transpose(1,2)
            pieces.append(x[batch[:,None],indices].flatten(1))
        pieces.append(windows['bar30m'].transpose(1,2)[:,self.anchor_indices].flatten(1))
        pieces.append(clock.to(pieces[0].dtype))
        return torch.cat(pieces,dim=-1)

    def finish_members(self,first):
        x=self.dropout(F.relu(first))
        for layer in self.layers[1:]:x=self.dropout(F.relu(layer(x)))
        return (torch.einsum('...mh,mho->...mo',x,self.head_weight.to(x.dtype))+
                self.head_bias.to(x.dtype)).float()

    def forward_member_heads(self,windows,clock):
        features=self.sample_features(windows,clock)
        return self.finish_members(self.layers[0](features[:,None,:]))

    def forward_heads(self,windows,clock):
        return self.forward_member_heads(windows,clock).mean(dim=-2)

    def forward(self,windows,clock):
        return self.forward_heads(windows,clock)[:,0]

    def forward_group_members(self,windows,latest_clock):
        recent,anchors,clocks=self.group_features(windows,latest_clock)
        return self.finish_members(self.layers[0].forward_group_parts(recent,anchors,clocks))

    def forward_group(self,windows,latest_clock):
        return self.forward_group_members(windows,latest_clock).mean(dim=-2)
