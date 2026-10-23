"""保留父77通道，用可信固定统计量补充近似数量绝对尺度。"""
import copy
import hashlib
import json

import torch
import numpy as np

import price_memory_inputs as parent
from price_geometry import FIELDS, PRICE_FIELDS

FREQUENCIES, DAILY_BARS, PREVIOUS_DAYS = parent.FREQUENCIES, parent.DAILY_BARS, parent.PREVIOUS_DAYS
QUANTITY_FIELDS = [f for f in FIELDS if f not in PRICE_FIELDS]
QUANTITY_INDICES = [FIELDS.index(f) for f in QUANTITY_FIELDS]
INPUT_CHANNELS = 121
NORMALIZATION = 'parent_price_memory_with_fixed_quantity_units'
CONTRACT = {'version':'fixed_training_quantity_units_v1','parent_view':'unchanged_77_price_memory_channels',
    'fields':QUANTITY_FIELDS,'inverse':'stored_z_times_fixed_std_plus_fixed_mean_fp32',
    'statistics':'frozen_training_per_stock_frequency_field','trust':'count_at_least_preprocessing_minimum_and_std_above_epsilon',
    'quantity_view':'clamp_reconstructed_log1p_at_zero_divide_10','untrusted':'new_view_and_flag_zero_parent_unchanged',
    'network_layout':'77_parent_22_quantity_units_22_validity','network_channels':INPUT_CHANNELS,
    'new_full_market_arrays':False}


def parent_configuration(config):
    result=copy.deepcopy(config)
    result['solution_id']='v2401'
    result['model']['input_channels']=77
    result['data'].pop('quantity_units',None)
    result['data']['normalization']=parent.NORMALIZATION
    return result


def validate_quantity_units_config(config):
    if config['data'].get('quantity_units')!=CONTRACT or config['data'].get('normalization')!=NORMALIZATION:
        raise ValueError('数量单位输入契约不一致')
    parent.validate_price_memory_config(parent_configuration(config))


def profiles_for_stocks(statistics, policy, stocks):
    """固定参数可信时保留均值/尺度；零尺度是补充视图不可用的标记。"""
    if statistics['fields']!=FIELDS:
        raise ValueError('固定统计字段顺序不同')
    result=np.zeros((len(stocks),len(QUANTITY_FIELDS),2),np.float32)
    for number,stock in enumerate(stocks):
        item=statistics['stocks'].get(str(stock))
        if item is None:continue
        count=np.asarray(item['count'],np.int64)
        mean=np.asarray(item['mean'],np.float64)
        m2=np.asarray(item['m2'],np.float64)
        if count.shape!=(36,) or mean.shape!=(36,) or m2.shape!=(36,) or (count<0).any() or (m2<0).any():
            raise ValueError('固定统计形状或数值不合法')
        scale=np.sqrt(np.divide(m2,count,out=np.zeros_like(m2),where=count>0))
        trusted=(count>=policy['min_observations']) & (scale>policy['scale_epsilon'])
        trusted=trusted[QUANTITY_INDICES]
        if not np.isfinite(mean[QUANTITY_INDICES][trusted]).all() or not np.isfinite(scale[QUANTITY_INDICES][trusted]).all():
            raise ValueError('可信统计量必须有限')
        result[number,:,0]=np.where(trusted,mean[QUANTITY_INDICES],0).astype(np.float32)
        result[number,:,1]=np.where(trusted,scale[QUANTITY_INDICES],0).astype(np.float32)
    return result


def profile_digest(profiles):
    digest=hashlib.sha256()
    for f in FREQUENCIES:
        values=np.ascontiguousarray(profiles[f],dtype='<f4')
        digest.update(f.encode());digest.update(json.dumps(list(values.shape)).encode());digest.update(values.tobytes())
    return digest.hexdigest()


def prepare_quantity_windows(raw, repairs, clock):
    result=parent.prepare_price_memory_windows({f:pair[:2] for f,pair in raw.items()},repairs,clock)
    for f in FREQUENCIES:
        codes,_,profiles=raw[f]
        if profiles.shape!=(len(clock),22,2) or not torch.isfinite(profiles).all() or (profiles[...,1]<0).any():
            raise ValueError('数量单位参数形状或数值不符')
        finite=torch.isfinite(codes)
        positions,exceptional=repairs[f]
        if positions.numel():finite.reshape(-1).index_copy_(0,positions,torch.isfinite(exceptional))
        cutoff=PREVIOUS_DAYS[f]*DAILY_BARS[f]+clock.argmax(dim=1)*(DAILY_BARS[f]//8)
        visible=torch.arange(codes.shape[1],device=codes.device)[None,:]<cutoff[:,None]
        valid=finite[:,:,QUANTITY_INDICES] & visible[:,:,None] & (profiles[:,None,:,1]>0)
        z=result[f][:,QUANTITY_INDICES,:].transpose(1,2)
        scaled=z*profiles[:,None,:,1].float()
        units=(scaled+profiles[:,None,:,0].float()).clamp_min(0)/10.
        units=torch.where(valid,units,0)
        extra=torch.cat((units,valid.float()),dim=2).transpose(1,2)
        result[f]=torch.cat((result[f],extra),dim=1).contiguous()
    return result
