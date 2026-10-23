"""父输入完整保留，固定数量统计只在取批时参与近似反变换。"""
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from quantity_units_inputs import (parent_configuration,validate_quantity_units_config,profiles_for_stocks,
                                    prepare_quantity_windows,FREQUENCIES)
from price_memory_inputs import PRECISION, STORAGE_COLUMNS as VALUE_COLUMNS
from v2401_data import (ProcessedStore as ParentStore,WindowBlock as ParentBlock,TradingCalendar,
                       regular_times,minute_slots,prediction_clock_values,section_indices,KEYS)


class WindowBlock(ParentBlock):
    def __init__(self,keys,grids,stocks,config,trading_days,profiles):
        validate_quantity_units_config(config)
        super().__init__(keys,grids,stocks,parent_configuration(config),trading_days)
        self.config=config
        if any(profiles[f].shape!=(len(stocks),22,2) for f in FREQUENCIES):
            raise ValueError('固定数量参数与股票网格不一致')
        self.profiles=profiles

    def batch(self,indices):
        indices=np.asarray(indices)
        requested,stocks=self.times[indices],self.stock_indices[indices]
        clock=prediction_clock_values(requested)
        raw,repairs={},{}
        for frequency,(times,values) in self.price_grids.items():
            stamps,positions=(item[self.day_indices[indices]] for item in self.plans[frequency])
            eligible=(positions>=0)&(stamps<=requested[:,None])
            if len(times):
                selected=values[positions.clip(0),stocks[:,None]].copy()
                selected[~eligible]=np.nan
            else:
                selected=np.full((len(indices),self.config['data']['window_bars'][frequency],73),np.nan,np.float32)
            codes=selected[...,:72].copy()
            repair_positions,repair_values=np.empty(0,np.int64),np.empty(0,np.float32)
            if self.config['data']['input_precision']==PRECISION:
                with np.errstate(over='ignore',invalid='ignore'):half=codes.astype(np.float16)
                overflow=np.isfinite(codes)&~np.isfinite(half)
                repair_positions=np.flatnonzero(overflow).astype(np.int64)
                repair_values=codes.reshape(-1)[repair_positions].copy()
                codes=half
            raw[frequency]=(torch.from_numpy(codes),torch.from_numpy(selected[...,72].copy()),
                            torch.from_numpy(self.profiles[frequency][stocks]))
            repairs[frequency]=(torch.from_numpy(repair_positions),torch.from_numpy(repair_values))
        result=prepare_quantity_windows(raw,repairs,torch.from_numpy(clock))
        return {f:v.numpy() for f,v in result.items()},clock


class ProcessedStore:
    def __init__(self,training_run,holdout_run,data_root,config,protocol):
        validate_quantity_units_config(config)
        self.parent=ParentStore(training_run,holdout_run,data_root,parent_configuration(config),protocol)
        self.config=config
        self.statistics={f:json.loads((Path(training_run)/'normalization'/(f+'.json')).read_text(encoding='utf-8')) for f in FREQUENCIES}
        preprocessing=json.loads((Path(training_run)/'config.json').read_text(encoding='utf-8'))
        self.normalization_policy=preprocessing['normalization']
        if any(pd.Timestamp(v['fit_interval']['stop'])>pd.Timestamp(protocol['splits']['train']['stop']) for v in self.statistics.values()):
            raise ValueError('固定统计量超过训练截止日')

    def __getattr__(self,name):
        return getattr(self.parent,name)

    def load(self,split,month):
        start=pd.Period(month,'M').start_time
        return self.load_range(split,start,(start.to_period('M')+1).start_time)

    def load_range(self,split,lo,hi):
        original=self.parent.load_range(split,lo,hi)
        stocks=pd.Index(sorted(original.keys.instrument.unique()))
        profiles={f:profiles_for_stocks(self.statistics[f],self.normalization_policy,stocks) for f in FREQUENCIES}
        return WindowBlock(original.keys,original.price_grids,stocks,self.config,self.days,profiles)
