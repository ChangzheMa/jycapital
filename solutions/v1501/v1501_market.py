"""按完整日期、完整当时股票池推理；股票分块仅用于序列编码。"""

import torch
import numpy as np

from v1501_grouped import StockDayIndex


def group_dates(block, grouping):
    rows = grouping.representatives
    if hasattr(block,'raw_batch'):
        return np.asarray(block.day)[rows]
    return block.keys.iloc[rows].date.to_numpy(dtype='datetime64[ns]').astype('datetime64[D]')


def predict_complete_days(model, block, device, config, indices=None, output=None):
    if model.training:
        raise ValueError('完整市场推理要求 eval 模式')
    grouping = StockDayIndex(block,indices)
    dates = group_dates(block,grouping)
    if output is None:
        output = np.full(len(block.keys),np.nan,dtype=np.float64)
    batch = config['model']['encoding_stock_batch_size']
    with torch.inference_mode():
        for day in np.unique(dates):
            groups = np.flatnonzero(dates==day)
            representatives = grouping.representatives[groups]
            parts = []
            if hasattr(block,'raw_batch'):
                from v1501_resident import ResidentBatches
                batches = ResidentBatches(block,representatives,block.targets,0,batch,device)
                for _,_,windows,clock,_ in batches:
                    parts.append(model.encode_group(windows,clock))
            else:
                for start in range(0,len(representatives),batch):
                    windows,clock = block.batch(representatives[start:start+batch])
                    parts.append(model.encode_group({key:torch.from_numpy(value).to(device) for key,value in windows.items()},
                                                    torch.from_numpy(clock).to(device)))
            membership = torch.from_numpy(grouping.samples[groups]>=0).to(device)
            predictions = model.decode_group(torch.cat(parts,dim=0),membership)[...,0].cpu().numpy()
            grouping.assign_predictions(output,groups,predictions)
    return output
