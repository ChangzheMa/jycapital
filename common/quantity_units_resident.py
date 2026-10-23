"""复用父数值数组，只在取批时加入约1MiB的固定数量参数表。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import numpy as np

from artifacts import read_json,write_json
import price_memory_resident as parent
from quantity_units_inputs import (FREQUENCIES,DAILY_BARS,PREVIOUS_DAYS,CONTRACT,parent_configuration,
    profiles_for_stocks,profile_digest,prepare_quantity_windows,validate_quantity_units_config)

FORMAT=parent.FORMAT
MINUTE_NS,DAY_NS,CLOCK_MINUTES=parent.MINUTE_NS,parent.DAY_NS,parent.CLOCK_MINUTES
require_memory=parent.require_memory


def validate_resident_config(config):
    validate_quantity_units_config(config)
    parent.validate_resident_config(parent_configuration(config))


def cache_key(store,manifest,transform):
    return parent.cache_key(store.parent,manifest,transform)


class QuantityBlock:
    def __init__(self,block,statistics,policy):
        self.parent=block
        self.profiles={f:profiles_for_stocks(statistics[f],policy,block.manifest['stocks']) for f in FREQUENCIES}
        self.profile_sha256=profile_digest(self.profiles)

    def __getattr__(self,name):
        return getattr(self.parent,name)

    def raw_batch(self,indices,destinations=None):
        indices=np.asarray(indices)
        original=None if destinations is None else {f:v[:2] for f,v in destinations.items()}
        raw,repairs,clock=self.parent.raw_batch(indices,original)
        result={}
        for f in FREQUENCIES:
            profiles=(self.profiles[f][self.parent.stock[indices]] if destinations is None else destinations[f][2][:len(indices)])
            if destinations is not None:np.take(self.profiles[f],self.parent.stock[indices],axis=0,out=profiles)
            result[f]=(*raw[f],profiles)
        return result,repairs,clock

    def batch(self,indices):
        raw,repairs,clock=self.raw_batch(indices)
        windows=prepare_quantity_windows({f:tuple(torch.from_numpy(v) for v in pair) for f,pair in raw.items()},
            {f:tuple(torch.from_numpy(v) for v in pair) for f,pair in repairs.items()},torch.from_numpy(clock))
        return {f:v.numpy() for f,v in windows.items()},clock

    def close(self):self.parent.close()


class ResidentBatches:
    def __init__(self,block,order,targets,start,batch_size,device):
        self.block,self.order,self.targets=block,order,targets
        self.start,self.batch_size,self.device=start,batch_size,torch.device(device)

    def __iter__(self):
        cuda=self.device.type=='cuda'
        stream=torch.cuda.Stream(device=self.device) if cuda else None
        slots=[]
        for _ in range(2):
            tensors={f:(torch.empty(self.batch_size,(PREVIOUS_DAYS[f]+1)*DAILY_BARS[f],72,dtype=torch.float16,pin_memory=cuda),
                torch.empty(self.batch_size,(PREVIOUS_DAYS[f]+1)*DAILY_BARS[f],dtype=torch.float32,pin_memory=cuda),
                torch.empty(self.batch_size,22,2,dtype=torch.float32,pin_memory=cuda)) for f in FREQUENCIES}
            slots.append({'tensors':tensors,'numpy':{f:tuple(v.numpy() for v in p) for f,p in tensors.items()},'event':None})
        offsets=iter(range(self.start,len(self.order),self.batch_size))
        def fill(offset,slot):
            if slot['event'] is not None:slot['event'].synchronize()
            selected=self.order[offset:offset+self.batch_size]
            raw,repairs,clock=self.block.raw_batch(selected,slot['numpy'])
            return offset,selected,slot,raw,repairs,clock
        def transfer(packed):
            offset,selected,slot,raw,repairs,clock=packed
            def prepare():
                values={f:tuple(v[:len(selected)].to(self.device,non_blocking=cuda) for v in slot['tensors'][f]) for f in FREQUENCIES}
                fixed={f:tuple(torch.from_numpy(v).to(self.device,non_blocking=cuda) for v in pair) for f,pair in repairs.items()}
                clocks=torch.from_numpy(clock).to(self.device,non_blocking=cuda)
                targets=torch.from_numpy(self.targets[selected]).to(self.device,non_blocking=cuda)
                return prepare_quantity_windows(values,fixed,clocks),clocks,targets
            if cuda:
                with torch.cuda.stream(stream):
                    values=prepare();ready=torch.cuda.Event();ready.record(stream)
                slot['event']=ready
            else:values,ready=prepare(),None
            return offset,selected,values,ready,packed
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                first=next(offsets,None)
                if first is None:return
                current=transfer(pool.submit(fill,first,slots[0]).result())
                number,offset=1,next(offsets,None)
                pending=pool.submit(fill,offset,slots[1]) if offset is not None else None
                while current is not None:
                    upcoming=transfer(pending.result()) if pending is not None else None
                    number+=1;offset=next(offsets,None)
                    pending=pool.submit(fill,offset,slots[number%2]) if offset is not None else None
                    position,selected,(windows,clock,target),ready,host=current
                    if cuda:
                        compute=torch.cuda.current_stream(self.device);compute.wait_event(ready)
                        for value in (*windows.values(),clock,target):value.record_stream(compute)
                    yield position,selected,windows,clock,target
                    current=upcoming
            finally:
                if stream is not None:stream.synchronize()


class ResidentStore(parent.ResidentStore):
    pass


def open_resident_store(store,run,transform,directory=None,report=None):
    previous=read_json(Path(run)/'resident-cache.json') if (Path(run)/'resident-cache.json').exists() else None
    original=parent.open_resident_store(store.parent,run,transform,directory,report)
    block=QuantityBlock(original.block,store.statistics,store.normalization_policy)
    if previous is not None and previous.get('quantity_profiles_sha256',block.profile_sha256)!=block.profile_sha256:
        raise ValueError('恢复时固定数量参数发生变化')
    record=read_json(Path(run)/'resident-cache.json')
    record.update(quantity_units=CONTRACT,quantity_profiles_sha256=block.profile_sha256,
        quantity_profile_bytes=sum(v.nbytes for v in block.profiles.values()),network_channels=121,
        additional_full_array_disk_bytes=0)
    write_json(Path(run)/'resident-cache.json',record)
    return ResidentStore(store,block,store.config['training']['resident_cache']['evaluation_days'])
