"""冻结EMA的时点主收益梯度：比较跨时点差异与同一时点的抽样噪声。"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for directory in ('common','solutions/v1601'):sys.path.insert(0,str(ROOT/directory))
from artifacts import now,read_json,sha256,write_json
from v1601_model import MultiFrequencyGRU
spec=importlib.util.spec_from_file_location('temporal_reader',ROOT/'scripts/audit-temporal-sensitivity.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)


def normalized_gram(matrix):
    values=matrix.double()
    norm=values.square().sum(dim=1).sqrt()
    if not torch.isfinite(values).all() or (norm<=1e-15).any():raise ValueError('梯度非有限或为零，不能解释余弦')
    return (values@values.T)/(norm[:,None]*norm[None,:])


def cross_half_profile(gram):
    if gram.shape!=(16,16):raise ValueError('需要8时点×2个独立股票半样本')
    a=torch.arange(0,16,2);b=a+1
    result=(gram[a[:,None],b[None,:]]+gram[b[:,None],a[None,:]])/2
    off=~torch.eye(8,dtype=torch.bool)
    return {'same_clock_cross_half_mean':float(result.diag().mean()),
            'different_clock_cross_half_mean':float(result[off].mean()),
            'same_clock_negative_fraction':float((result.diag()<0).double().mean()),
            'different_clock_negative_fraction':float((result[off]<0).double().mean()),
            'cross_half_clock_cosines':result.numpy().tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/clock-gradient-alignment-20261023.json')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20261023);torch.use_deterministic_algorithms(True)
    run=ROOT/'outputs/runs/v1601-price-geometry-s42';selection=read_json(run/'selection.json')
    if sha256(run/'best.pt')!=selection['model_sha256']:raise ValueError('权重指纹改变')
    config=read_json(run/'config.json');source=read_json(run/'manifest.json')['source']['files']
    for path in ('solutions/v1601/v1601_model.py','solutions/v1601/v1601_schema.py','common/geometry_resident.py'):
        if sha256(ROOT/path)!=source[path]:raise ValueError('冻结推理源码改变')
    model=MultiFrequencyGRU(config).eval();model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    parameters=list(model.named_parameters());groups={
        'encoder':[i for i,(name,_) in enumerate(parameters) if name.startswith('encoders.')],
        'fusion_and_readout':[i for i,(name,_) in enumerate(parameters) if not name.startswith('encoders.')]}
    cache=ROOT/'outputs/preprocessing/resident-cache'/reader.CACHE_KEY
    block=reader.read_only_block(cache)
    if sha256(cache/'targets.npy')!=block.manifest['files']['targets.npy']['sha256']:raise ValueError('训练目标指纹改变')
    target=np.load(cache/'targets.npy',mmap_mode='r',allow_pickle=False)
    days=block.arrays['days'].astype('datetime64[ns]').astype('datetime64[D]').astype(str)
    fit=read_json(ROOT/'experiments/frozen-fit-gap-20261023.json')
    selected=[]
    for year in range(2020,2024):
        selected.extend([r['day'] for r in fit['training_date_selection'] if r['day'].startswith(str(year))][:2])
    records=[]
    for stamp in selected:
        day=int(np.flatnonzero(days==stamp)[0]);rows=np.flatnonzero(block.day==day)
        stocks=np.unique(block.stock[rows])
        chosen=np.random.default_rng(np.random.SeedSequence([20261023,4,day])).choice(stocks,64,replace=False)
        mapping=np.full((len(block.manifest['stocks']),8),-1,np.int64)
        mapping[block.stock[rows],block.clock[rows]]=rows
        indices=mapping[chosen];latest=7-(indices>=0)[:,::-1].argmax(axis=1)
        representatives=indices[np.arange(64),latest]
        windows,clock=block.batch(representatives)
        predictions=model.forward_group({f:torch.from_numpy(x) for f,x in windows.items()},torch.from_numpy(clock))[...,0]
        labels=torch.from_numpy(np.where(indices>=0,target[indices.clip(0)],np.nan))
        matrices={name:[] for name in groups};counts=[]
        for clock_index in range(8):
            for half in range(2):
                ids=torch.arange(half*32,(half+1)*32)
                valid=torch.isfinite(labels[ids,clock_index]);ids=ids[valid]
                if len(ids)<16:raise ValueError('有效半样本不足16，不能解释梯度差异')
                loss=F.huber_loss(predictions[ids,clock_index].double(),labels[ids,clock_index],delta=2.)
                gradients=torch.autograd.grad(loss,[p for _,p in parameters],retain_graph=(clock_index,half)!=(7,1),allow_unused=True)
                for name,positions in groups.items():
                    values=[torch.zeros_like(parameters[i][1]).reshape(-1) if gradients[i] is None else gradients[i].detach().reshape(-1) for i in positions]
                    matrices[name].append(torch.cat(values))
                counts.append(len(ids))
        profiles={}
        full_gram=torch.zeros((16,16),dtype=torch.float64)
        for name,values in matrices.items():
            matrix=torch.stack(values)
            profiles[name]=cross_half_profile(normalized_gram(matrix))
            double=matrix.double();full_gram+=double@double.T
            del matrix,double
        norms=full_gram.diag().sqrt()
        profiles['all_parameters']=cross_half_profile(full_gram/(norms[:,None]*norms[None,:]))
        records.append({'day':stamp,'stock_days':64,'half_valid_counts':counts,
            'sample_rows_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),'profiles':profiles})
        print({'day':stamp,'encoder_same_clock':profiles['encoder']['same_clock_cross_half_mean'],
               'encoder_different_clock':profiles['encoder']['different_clock_cross_half_mean']},flush=True)
        del predictions,gradients,matrices,windows,labels
    summaries={}
    for name in ('encoder','fusion_and_readout','all_parameters'):
        summaries[name]={key:np.asarray([r['profiles'][name][key] for r in records]).mean(axis=0).tolist()
                         for key in records[0]['profiles'][name]}
    report={'computed_at':now(),'script_sha256':sha256(__file__),
       'reader_script_sha256':sha256(ROOT/'scripts/audit-temporal-sensitivity.py'),
       'date_source_sha256':sha256(ROOT/'experiments/frozen-fit-gap-20261023.json'),
       'run_id':run.name,'model_sha256':selection['model_sha256'],'clock_order':list(reader.CLOCKS),
       'sampling':'此前固定训练抽样日期中，每年前两个日期，共8日；每日期独立随机64只股票，随机顺序分成32+32两个不重叠半样本。',
       'objective':'已提供收益标签的冻结截面zscore，主收益Huber delta=2；不含排名和专家辅助项；固定EMA、eval模式禁用Dropout。',
       'method':'分别求8时点的两个半样本梯度。比较同一时点跨半样本的余弦与不同时间点跨半样本余弦；不将不同股票的抽样噪声直接解释为时点冲突。',
       'model_training_performed':False,'optimizer_created':False,'weight_updates':0,'exposure_read':False,
       'validation_or_test_read':False,'summaries':summaries,'dates':records,
       'limitations':['8个训练日是局部诊断，不能证明分时网络会改善IC。',
                      '在固定EMA权重上测主收益梯度，不等同于正式训练过程中含Dropout、排名和专家辅助项的完整梯度。',
                      '梯度方向会随参数化、当前拟合状态、标签噪声和市场日期改变；负余弦不能单独证明需要任务拆分。',
                      '没有更新权重、拟合新模型或改变正式选模与评测口径。']}
    write_json(args.output,report)


if __name__=='__main__':main()
