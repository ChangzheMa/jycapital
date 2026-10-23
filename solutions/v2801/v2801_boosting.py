"""量化树训练、验证IC选迭代次数和可恢复状态；推理使用便携Torch树。"""
import gc
import json
import os
from pathlib import Path
import time

import torch
import numpy as np
import pandas as pd
import xgboost as xgb

from artifacts import read_json, write_json, sha256, now
from early_stopping import select_epoch, should_stop, patience_epochs
from baseline_evaluation import evaluate
from v2401_resident import require_memory
from v2801_trees import TreeEnsemble

FORMAT = 'jycapital-neural-tree-checkpoint-v1'


class TrainingIterator(xgb.DataIter):
    def __init__(self, cache, batch_rows):
        self.cache, self.batch_rows = cache, batch_rows
        self.offset = 0
        super().__init__(release_data=True)

    def reset(self):
        self.offset = 0

    def next(self, input_data):
        arrays = self.cache.arrays
        while self.offset < len(arrays['primary']):
            start = self.offset
            self.offset = min(start+self.batch_rows,len(arrays['primary']))
            y = np.asarray(arrays['primary'][start:self.offset])
            valid = np.isfinite(y)
            if not valid.any():continue
            features = np.array(arrays['features'][start:self.offset][valid],dtype=np.float32,copy=True)
            weights = np.array(arrays['weights'][start:self.offset][valid],dtype=np.float32,copy=True)
            if not np.isfinite(features).all() or not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('树训练表示或日期权重非法')
            input_data(data=features,label=y[valid].astype(np.float32),weight=weights,base_margin=features[:,-1].copy())
            return True
        return False


def make_matrix(cache, config):
    rows = cache.manifest['finite_primary_rows']
    width = cache.manifest['feature_dim']
    # 预算留出量化矩阵、构建索引和当前浮点批；不再保留完整父行情。
    estimate = rows*width*4 + config['training']['feature_batch_rows']*width*8 + rows*32
    require_memory(estimate,config['training']['resident_cache']['memory_reserve_gib'])
    iterator = TrainingIterator(cache,config['training']['feature_batch_rows'])
    matrix = xgb.QuantileDMatrix(iterator,max_bin=config['training']['booster']['max_bin'],
                                nthread=config['training']['cpu_threads'])
    if matrix.num_row() != rows or matrix.num_col() != width:
        raise ValueError('量化矩阵的有效标签行数或维度不符')
    return matrix


@torch.inference_mode()
def predict_cache(trees, cache, batch_rows, device):
    device = torch.device(device)
    trees = trees.to(device).eval()
    source = cache.arrays['features']
    output = np.empty(len(source),np.float64)
    for start in range(0,len(source),batch_rows):
        values = torch.from_numpy(np.array(source[start:start+batch_rows],copy=True)).to(device)
        prediction = values[:,-1] + trees(values)
        if not torch.isfinite(prediction).all():
            raise ValueError('树补充预测非有限')
        output[start:start+len(values)] = prediction.cpu().numpy()
    return output


def evaluate_cache(trees, cache, config, protocol, device):
    keys = pd.read_parquet(cache.directory/'keys.parquet')
    prediction = predict_cache(trees,cache,config['inference']['feature_batch_rows'],device)
    frame = keys[['date','instrument']].copy()
    frame['factor'] = prediction
    result = evaluate(frame,keys,protocol)
    primary = cache.arrays['primary']
    valid = np.isfinite(primary)
    loss = float(np.square(prediction[valid]-primary[valid]).mean())
    return frame,result,loss


def save_evaluation(run, split, frame, result):
    metrics, sections, monthly = result
    directory = Path(run)/'evaluation'/split
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'metrics.json',metrics)
    frame.to_parquet(directory/'predictions.parquet',index=False)
    sections.to_parquet(directory/'cross-sections.parquet',index=False)
    monthly.to_csv(directory/'monthly.csv',index=False,encoding='utf-8-sig')


def checkpoint(path, booster, trees, encoder_state, config, parent_config, state, history, rounds, identities):
    payload = {'format':FORMAT,'config':config,'parent_config':parent_config,
        'state_dict':{**{'encoder.'+k:v.detach().cpu() for k,v in encoder_state.items()},
                      **{'trees.'+k:v.detach().cpu() for k,v in trees.state_dict().items()}},
        'epoch':len(history),'boosting_rounds':rounds,'selected_tree_count':int(trees.active_trees.item()),
        'weight_source':'frozen_parent_ema_plus_boosted_trees','parent_model_sha256':config['model']['parent_model_sha256'],
        'booster_json':booster.save_raw(raw_format='json').decode('utf-8'),
        'booster_training_config':json.loads(booster.save_config()),'training_device':str(trees.condition.device),
        'selection_state':dict(state),'history':list(history),'feature_identities':identities,
        'xgboost_version':xgb.__version__}
    temporary = Path(str(path)+'.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def fit_booster(train, validation, config, parent_config, encoder_state, protocol, run, device,
                *, resume=False, report=None, after_update=None):
    if xgb.__version__ != '3.2.0':
        raise ValueError('本候选要求已验证的XGBoost3.2.0')
    run = Path(run)
    params = dict(config['training']['booster'],device=str(device))
    identities = {'train':sha256(train.directory/'manifest.json'),'validation':sha256(validation.directory/'manifest.json')}
    started = time.perf_counter()
    matrix = make_matrix(train,config)
    if report:report('tree_quantile_matrix_ready',seconds=time.perf_counter()-started,rows=matrix.num_row(),features=matrix.num_col())
    trees = TreeEnsemble(config['model']['feature_dim'],config['training']['boosting_rounds'],params['max_depth']).to(device)
    booster = xgb.Booster(params=params,cache=[matrix])
    history, state, rounds = [], {'best':-float('inf'),'best_epoch':0,'stale':0}, 0
    if resume:
        saved = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if saved['format'] != FORMAT or saved['config'] != config or saved['parent_config'] != parent_config or saved['feature_identities'] != identities or saved['xgboost_version'] != xgb.__version__:
            raise ValueError('树恢复配置、表示来源或库版本改变')
        previous_device,current_device=torch.device(saved['training_device']),torch.device(device)
        if (previous_device.type,previous_device.index or 0)!=(current_device.type,current_device.index or 0):
            raise ValueError('精确树恢复要求相同训练设备')
        for name,value in encoder_state.items():
            if not torch.equal(value.cpu(),saved['state_dict']['encoder.'+name]):
                raise ValueError('恢复时父编码器权重改变')
        booster.load_model(bytearray(saved['booster_json'].encode()))
        booster.set_param(params)
        state, history, rounds = saved['selection_state'],saved['history'],saved['boosting_rounds']
        trees.import_booster(saved['booster_json'])
        if booster.num_boosted_rounds() != rounds:
            raise ValueError('树恢复迭代游标不一致')
        pd.DataFrame(history).to_csv(run/'history.csv',index=False,encoding='utf-8-sig')
    else:
        frame,result,loss = evaluate_cache(trees,validation,config,protocol,device)
        if result[0]['ic_undefined_cross_sections']:
            raise ValueError('父初始预测存在未定义IC截面')
        state.update(best=result[0]['ic_mean'])
        write_json(run/'initial-validation.json',{'epoch':0,'tree_count':0,'metrics':result[0],
            'validation_primary_mse_loss':loss,'source':'frozen_parent_prediction'})
        save_evaluation(run,'validation',frame,result)
        checkpoint(run/'best.pt',booster,trees,encoder_state,config,parent_config,state,history,rounds,identities)
        checkpoint(run/'last.pt',booster,trees,encoder_state,config,parent_config,state,history,rounds,identities)
    session_history_start=len(history)
    for epoch in range(len(history)+1,config['training']['epochs']+1):
        if should_stop(state):break
        started = time.perf_counter()
        stop = min(epoch*config['training']['evaluation_rounds'],config['training']['boosting_rounds'])
        if rounds >= stop:break
        # 行列抽样器的隐藏状态不由模型JSON完整恢复。每个验证块都使用相同的
        # 显式重建与派生种子，正常运行和恢复走同一路径，保留预设随机正则化。
        block_params = dict(params, seed=config['seed']+epoch)
        previous = booster.save_raw(raw_format='json')
        booster = xgb.Booster(params=block_params,cache=[matrix],model_file=previous)
        booster.set_param(block_params)
        for iteration in range(rounds,stop):
            booster.update(matrix,iteration)
            if after_update:after_update(iteration+1)
        rounds = stop
        trees.import_booster(booster.save_raw(raw_format='json'))
        training_eval = booster.eval_set([(matrix,'train')],iteration=rounds-1)
        rmse = float(training_eval.rsplit(':',1)[1])
        frame,result,loss = evaluate_cache(trees,validation,config,protocol,device)
        metrics = result[0]
        eligible = metrics['ic_undefined_cross_sections'] == 0 and metrics['coverage'] == 1.
        improved = select_epoch(state,metrics['ic_mean'],epoch,eligible)
        row = {'epoch':epoch,'boosting_rounds':rounds,'train_loss':rmse**2,'train_primary_mse_loss':rmse**2,
            'validation_loss':loss,'validation_primary_mse_loss':loss,'validation_ic_mean':metrics['ic_mean'],
            'validation_rank_ic_mean':metrics['rank_ic_mean'],'ic_mean':metrics['ic_mean'],'rank_ic_mean':metrics['rank_ic_mean'],
            'best_epoch':state['best_epoch'],'improved':improved,'selection_eligible':eligible,
            'stale_epochs':state['stale'],'patience_epochs':patience_epochs(state['best_epoch']),
            'model_undefined_sections':metrics['ic_undefined_cross_sections'],
            'train_rows':matrix.num_row(),'seconds':time.perf_counter()-started}
        history.append(row)
        if improved:
            save_evaluation(run,'validation',frame,result)
            checkpoint(run/'best.pt',booster,trees,encoder_state,config,parent_config,state,history,rounds,identities)
        checkpoint(run/'last.pt',booster,trees,encoder_state,config,parent_config,state,history,rounds,identities)
        pd.DataFrame(history).to_csv(run/'history.csv',index=False,encoding='utf-8-sig')
        if report:report('tree_validation_finished',**row)
    del matrix,booster,trees
    gc.collect()
    if torch.device(device).type == 'cuda':torch.cuda.empty_cache()
    selected = torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    selection = {'best_epoch':state['best_epoch'],'best_validation_metric':state['best'],
        'best_validation_ic':state['best'],'selection_metric':'ic_mean','epochs_completed':len(history),
        'epoch_unit':'25_boosting_rounds' if config['training']['evaluation_rounds']==25 else 'configured_boosting_block',
        'selected_boosting_rounds':selected['boosting_rounds'],'boosting_rounds_completed':rounds,
        'evaluation_weights':'frozen_parent_ema_plus_boosted_trees','early_stopping':config['training']['early_stopping'],
        'patience_epochs':patience_epochs(state['best_epoch']),'stale_epochs':state['stale'],
        'model_sha256':sha256(run/'best.pt'),'selected_at':now(),
        'training_seconds':sum(r['seconds'] for r in history),
        'training_seconds_this_session':sum(r['seconds'] for r in history[session_history_start:]),
        'training_timing_scope':'boosting_blocks_including_validation_excludes_feature_extraction_and_quantile_construction',
        'stop_reason':'early_stopping' if should_stop(state) else 'maximum_boosting_rounds'}
    write_json(run/'selection.json',selection)
    return selected,selection
