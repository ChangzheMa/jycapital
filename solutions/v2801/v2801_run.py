"""冻结历史表示、量化树训练及完整双集评估的运行编排。"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (ROOT/'common', ROOT/'evaluation'):
    sys.path.insert(0,str(directory))
from artifacts import now, read_json, write_json, sha256, environment, snapshot_sources, log_event
from research_queue import validate_spec, readiness_error, source_files
from early_stopping import validate_early_stopping
from protocol_compatibility import validate_protocol_compatibility
from training_resources import record_resources
import v2401_train as parent_train
import price_memory_resident as resident
import price_memory_evaluation_cache as evaluation_cache
from price_memory_inputs import base_data_manifest
from v2401_data import ProcessedStore
from v2401_text_artifact import state_digest
from v2801_features import FrozenHistoryEncoder
from v2801_cache import FeatureCache, cache_identity, build_features
from v2801_boosting import fit_booster, evaluate_cache, save_evaluation
from v2801_model import MultiFrequencyNeuralTree


def validate_training(config, protocol):
    if config.get('solution_id') != 'v2801' or config['model']['family'] != 'neural_boosted_trees':
        raise ValueError('v2801模型契约不一致')
    parent = config['backbone_config']
    parent_train.validate_training(parent,protocol)
    if config['data'] != parent['data'] or config['model']['feature_dim'] != parent['model']['fusion_hidden_dim']+240:
        raise ValueError('父输入或624维表示布局改变')
    if config['model']['feature_layout'] != 'fusion_384_latest_grid_3x77_clock8_parent_main1':
        raise ValueError('学习表示布局不一致')
    training = config['training']
    validate_early_stopping(training)
    if training['resident_cache']['memory_reserve_gib'] != 2.0:
        raise ValueError('v2801要求2GiB主机预留')
    if protocol['selection_metric'] != 'ic_mean':
        raise ValueError('v2801按验证IC选树迭代次数')
    for field in ('epochs','evaluation_rounds','boosting_rounds','feature_batch_rows','stock_day_batch_size','cpu_threads'):
        if type(training.get(field)) is not int or training[field] < 1:
            raise ValueError('训练预算必须为正整数: '+field)
    if training['epochs']*training['evaluation_rounds'] != training['boosting_rounds']:
        raise ValueError('树迭代次数与验证块预算不一致')
    if training['sampling_seed_policy'] != 'fresh_booster_per_validation_block_seed_plus_epoch':
        raise ValueError('树抽样恢复策略不一致')
    booster = training['booster']
    for field,value in {'objective':'reg:squarederror','booster':'gbtree','tree_method':'hist','base_score':0.,'seed_per_iteration':True}.items():
        if booster.get(field) != value:raise ValueError('树训练契约不一致: '+field)
    if booster['seed'] != config['seed'] or booster['device'] not in ('cpu','cuda') or booster['nthread'] != training['cpu_threads']:
        raise ValueError('树种子、设备或线程配置不一致')
    if not 1 <= booster['max_depth'] <= 10 or not 2 <= booster['max_bin'] <= 256:
        raise ValueError('树深度或量化预算越界')
    if not 0 < booster['subsample'] <= 1 or not 0 < booster['colsample_bytree'] <= 1:
        raise ValueError('树抽样概率非法')
    if config['inference']['factor_scale'] != 1. or training['temporal_weighting'] != parent['training']['temporal_weighting']:
        raise ValueError('主预测单位或日期权重定义改变')


def input_directory(store, source, split):
    if split == 'train':
        key = resident.cache_key(store,source,parent_train.prepare_training_targets)
    else:
        base_key = evaluation_cache.base_evaluation.cache_key(store.base,split,base_data_manifest(source))
        key = resident.side_key(base_key)
    return store.training_run.parent/'price-memory-cache'/key


def extract_split(store, encoder, config, source, split, run, cache_root, mode):
    directory = input_directory(store,source,split)
    record_path = Path(run)/'feature-caches.json'
    records = read_json(record_path) if record_path.exists() else {}
    report = lambda event,**fields:log_event(run,event,**fields)
    if (directory/'manifest.json').is_file():
        key,identity = cache_identity(config,source,read_json(directory/'manifest.json'),split)
        destination = Path(cache_root)/key
        if destination.exists():
            result = FeatureCache(destination,identity)
            records[split] = {'path':str(destination.resolve()),'manifest_sha256':sha256(destination/'manifest.json'),
                             'cache_key':key,'input_cache_manifest':str(directory/'manifest.json')}
            write_json(record_path,records)
            return result
    started = time.perf_counter()
    wrapper = None
    block = None
    try:
        if split == 'train':
            wrapper = resident.open_resident_store(store,run,parent_train.prepare_training_targets,report=report)
            block,keys = wrapper.block,None
        else:
            wrapper = evaluation_cache.open_evaluation_data(store,split,source,mode=mode,report=report)
            block,keys = wrapper.block,wrapper.keys
        key,identity = cache_identity(config,source,block.manifest,split)
        destination = Path(cache_root)/key
        build_features(destination,identity,block,encoder,config,store.protocol,split,keys=keys,report=report)
        records[split] = {'path':str(destination.resolve()),'manifest_sha256':sha256(destination/'manifest.json'),
                         'cache_key':key,'input_cache_manifest':str(directory/'manifest.json')}
        write_json(record_path,records)
    finally:
        if block is not None:block.close()
        block = wrapper = None
        gc.collect()
        if next(encoder.parameters()).device.type == 'cuda':torch.cuda.empty_cache()
    report('neural_features_ready',split=split,seconds=time.perf_counter()-started,
           rows=read_json(destination/'manifest.json')['rows'])
    return FeatureCache(destination,identity)


def execute(args):
    config,protocol = read_json(args.config),read_json(args.protocol)
    validate_training(config,protocol)
    if xgb.__version__ != '3.2.0':raise ValueError('要求XGBoost3.2.0')
    if not args.training_confirmed:raise ValueError('正式训练须具有明确授权')
    run = args.run_dir.resolve()
    if not args.reproduction_only:
        spec = validate_spec(ROOT,HERE/'experiment.json')
        if spec['status'] != 'ready' or spec['run_id'] != run.name or readiness_error(ROOT,spec):
            raise ValueError('候选尚未就绪或发布指纹改变')
    parent_run = args.parent_run.resolve() if args.parent_run else ROOT/'outputs/runs'/config['parent_run_id']
    parent_manifest = read_json(parent_run/'manifest.json')
    if parent_manifest['status'] != 'complete' or sha256(parent_run/'best.pt') != config['model']['parent_model_sha256']:
        raise ValueError('父运行或所选EMA权重改变')
    parent_config = config['backbone_config']
    if parent_config != read_json(parent_run/'config.json'):
        raise ValueError('父模型配置与冻结权重不对应')
    validate_protocol_compatibility(protocol,read_json(parent_run/'protocol.json'))
    parent_checkpoint = torch.load(parent_run/'best.pt',map_location='cpu',weights_only=True)
    if parent_checkpoint.get('weight_source') != 'ema':raise ValueError('父模型必须使用冻结选中的EMA')
    if parent_checkpoint['config'] != parent_config:
        raise ValueError('父checkpoint内配置与声明不同')
    if not args.reproduction_only and parent_manifest['selection']['model_sha256'] != config['model']['parent_model_sha256']:
        raise ValueError('引用权重不是父版正式选中权重')
    if args.resume:
        manifest = read_json(run/'manifest.json')
        if manifest['status'] == 'complete':raise ValueError('已完成运行不重复训练')
        if read_json(run/'config.json') != config or read_json(run/'protocol.json') != protocol:
            raise ValueError('恢复时配置或协议改变')
        for path,digest in manifest['source']['files'].items():
            if sha256(run/'source'/path) != digest:raise ValueError('冻结源码改变: '+path)
        for path,digest in source_files(ROOT,'v2801').items():
            if manifest['source']['files'].get(path) != digest:raise ValueError('恢复入口与冻结源码不同: '+path)
    else:
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/'config.json',config)
        write_json(run/'protocol.json',protocol)
        manifest = {'run_id':run.name,'solution_id':'v2801','parent_solution_id':'v2401',
            'parent_run_id':config['parent_run_id'],'status':'running','stage':'feature_extraction','created_at':now(),
            'formal_attempts':1,'model_family':'neural_boosted_trees','training_kind':'boosting_on_frozen_neural_features',
            'source':snapshot_sources(ROOT,run,'v2801')}
        write_json(run/'manifest.json',manifest)
    device = torch.device(config['training']['booster']['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():raise RuntimeError('正式配置要求CUDA，未自动切换训练设备')
    torch.set_num_threads(config['training']['cpu_threads'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.use_deterministic_algorithms(config['training']['deterministic'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = FrozenHistoryEncoder(parent_config).to(device)
    encoder.load_state_dict(parent_checkpoint['state_dict'])
    parent_digest = state_digest(encoder.state_dict())
    env = environment()
    env.update(device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        parameters=sum(p.numel() for p in encoder.parameters()),neural_parameters=sum(p.numel() for p in encoder.parameters()),
        model_family='neural_boosted_trees',xgboost=xgb.__version__,tree_build_info=xgb.build_info(),
        parameter_count_definition='frozen_neural_parameters_plus_selected_tree_thresholds_and_leaf_values',
        evaluation_precision='float32',input_precision=parent_config['data']['input_precision'],
        configured_host_memory_reserve_gib=2.,gpu_memory_measurement_scope='torch_allocator_excludes_native_xgboost_allocations')
    write_json(run/'environment.json',env)
    (run/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in env['packages'].items())+'\n',encoding='utf-8')
    log_event(run,'training_resumed' if args.resume else 'training_started',parameters=env['parameters'],
              model_family='neural_boosted_trees',formal_attempts=1)
    record_resources(run,device)
    store = ProcessedStore(args.training_run,args.holdout_run,args.data_root,parent_config,protocol)
    expected_source = read_json(parent_run/'data-manifest.json')
    if not (run/'data-manifest.json').exists():parent_train.data_manifest(store,run)
    source = read_json(run/'data-manifest.json')
    if source != expected_source:raise ValueError('新运行与冻结父版的输入、标签或预处理来源不一致')
    cache_root = args.feature_cache_root or args.training_run.parent/'neural-feature-cache'
    train = validation = test = None
    try:
        trees_already_selected = args.resume and manifest.get('stage')=='test_evaluation'
        if trees_already_selected:
            selection=read_json(run/'selection.json')
            if sha256(run/'best.pt')!=selection['model_sha256']:
                raise ValueError('恢复测试时选定权重改变')
            selected=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            if selected['config']!=config:raise ValueError('恢复测试时权重配置不同')
        else:
            train = extract_split(store,encoder,config,source,'train',run,cache_root,args.evaluation_cache_mode)
            validation = extract_split(store,encoder,config,source,'validation',run,cache_root,args.evaluation_cache_mode)
            encoder.cpu()
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            manifest.update(stage='tree_training')
            write_json(run/'manifest.json',manifest)
            report = lambda event,**fields:log_event(run,event,**fields)
            selected,selection = fit_booster(train,validation,config,parent_config,encoder.state_dict(),protocol,run,device,
                resume=args.resume and (run/'last.pt').exists(),report=report)
            train.close();train=None
            validation.close();validation=None
        manifest.update(stage='test_evaluation',selection=selection)
        write_json(run/'manifest.json',manifest)
        encoder.to(device)
        test = extract_split(store,encoder,config,source,'test',run,cache_root,args.evaluation_cache_mode)
        model = MultiFrequencyNeuralTree(config).to(device)
        model.load_state_dict(selected['state_dict'])
        frame,result,test_loss = evaluate_cache(model.trees,test,config,protocol,device)
        save_evaluation(run,'test',frame,result)
        if state_digest(encoder.state_dict()) != parent_digest or sha256(parent_run/'best.pt') != config['model']['parent_model_sha256']:
            raise ValueError('冻结父权重在运行期间改变')
        trees = json.loads(selected['booster_json'])['learner']['gradient_booster']['model']['trees']
        env['tree_numeric_parameters'] = sum(len(tree['split_conditions']) for tree in trees)
        env['parameters'] = env['neural_parameters']+env['tree_numeric_parameters']
        env['selected_tree_count'] = len(trees)
        write_json(run/'environment.json',env)
        write_json(run/'loss-consistency.json',{'loss':'weighted_primary_zscore_squared_error','training_weights':'frozen_parent_ema_plus_boosted_trees',
            'provided_label_unchanged':True,'label_transform':'parent_cross_section_zscore','xgboost_label_precision':'float32',
            'training_loss_reduction':'global_weighted_sample_mean','evaluation_loss_reduction':'unweighted_finite_primary_sample_mean',
            'formal_selection_metric':'equal_cross_section_ic_mean','test_primary_mse_loss':test_loss,
            'parent_weight_sha256':config['model']['parent_model_sha256'],'parent_encoder_unchanged':True})
        parent_train.draw_history(run,pd.read_csv(run/'history.csv').to_dict('records'))
        manifest.update(status='complete',stage='complete',completed_at=now(),selection=selection,
            metrics={split:read_json(run/'evaluation'/split/'metrics.json') for split in ('validation','test')},
            feature_caches=read_json(run/'feature-caches.json'),limitations=[
                '一次seed42树训练，父网络使用本仓库已选中的EMA；测试区间已被持续研究观察。',
                '神经表示固定，平方误差树补充与原神经Huber目标不同；不把训练损失直接排名。',
                '每25棵树为一个验证块，块边界重新创建提升器并派生种子，保证抽样恢复路径一致；包含额外预测缓存重建成本。',
                'GPU资源记录中的Torch峰值不包含XGBoost原生分配；需另由训练监控记录总显存。'])
        record_resources(run,device)
        write_json(run/'manifest.json',manifest)
        log_event(run,'training_complete',selection=selection)
        return manifest
    finally:
        for cache in (train,validation,test):
            if cache is not None:cache.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config',type=Path,default=HERE/'model-config.json')
    result.add_argument('--protocol',type=Path,default=ROOT/'evaluation/protocols/local-v2-ic.json')
    result.add_argument('--training-run',type=Path,default=ROOT/'outputs/preprocessing/v2-20261010')
    result.add_argument('--holdout-run',type=Path,default=ROOT/'outputs/preprocessing/v2-holdout-20261011')
    result.add_argument('--data-root',type=Path,default=ROOT/'data')
    result.add_argument('--run-dir',type=Path,required=True)
    result.add_argument('--parent-run',type=Path)
    result.add_argument('--feature-cache-root',type=Path)
    result.add_argument('--resident-cache',type=Path,help='兼容调度参数；按父输入身份复用实际缓存')
    result.add_argument('--evaluation-cache-mode',choices=('auto','resident','stream'),default='stream')
    result.add_argument('--training-confirmed',action='store_true')
    result.add_argument('--reproduction-only',action='store_true')
    result.add_argument('--resume',action='store_true')
    return result


def main():
    result = execute(parser().parse_args())
    print(json.dumps({'run_id':result['run_id'],'status':result['status'],'selection':result['selection']},ensure_ascii=False))
