"""只用合成数据验证父编码批量、量化树更新和便携推理；不启动正式训练。"""
import copy
import json
import time

import torch
import numpy as np
import xgboost as xgb

from v2401_train import preflight_inputs
from v2801_model import MultiFrequencyNeuralTree


def compare_resume_models(uninterrupted, resumed):
    """预测结构严格相同；GPU增益统计的末位舍入单独核验。"""
    left, right = copy.deepcopy(uninterrupted), copy.deepcopy(resumed)
    left_trees, right_trees = left['model']['trees'], right['model']['trees']
    if len(left_trees) != len(right_trees):
        raise ValueError('同设备分块抽样恢复后树数量不一致')
    maximum_gain_difference = 0.
    for first, second in zip(left_trees, right_trees):
        first_gain, second_gain = first.pop('loss_changes'), second.pop('loss_changes')
        if len(first_gain) != len(second_gain):
            raise ValueError('恢复后的增益统计布局不一致')
        a, b = np.asarray(first_gain, np.float64), np.asarray(second_gain, np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('恢复后的增益统计非有限')
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        maximum_gain_difference = max(maximum_gain_difference, float(np.max(np.abs(a-b), initial=0.)))
    if left != right:
        raise ValueError('同设备分块抽样恢复后预测结构或其他树状态不一致')
    return {'native_tree_document_exact': uninterrupted == resumed,
            'prediction_structure_and_other_state_exact': True,
            'nonpredictive_gain_max_abs_difference': maximum_gain_difference,
            'nonpredictive_gain_tolerance': {'rtol':1e-6, 'atol':1e-7}}


def run_device_preflight(config, device):
    device = torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    torch.set_num_threads(config['training']['cpu_threads'] if device.type=='cuda' else 1)
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    model=MultiFrequencyNeuralTree(config).to(device).eval()
    train_batch=config['training']['stock_day_batch_size'] if device.type=='cuda' else 2
    inference_batch=config['inference']['stock_day_batch_size'] if device.type=='cuda' else 4
    windows,clock=preflight_inputs(config['backbone_config'],train_batch,device)
    started=time.perf_counter()
    with torch.inference_mode():
        features,_=model.encoder.group_features(windows,clock)
    if device.type=='cuda':torch.cuda.synchronize()
    encoding_seconds=time.perf_counter()-started
    values=features.reshape(-1,features.shape[-1]).cpu().numpy()
    # 量化阶段独立于原始窗口；使用足够样本让生产的最小叶权重参与检查。
    repeats=max(1,(2048+len(values)-1)//len(values))
    values=np.tile(values,(repeats,1)).astype(np.float32)
    rng=np.random.default_rng(config['seed'])
    values=values+rng.normal(0,.01,values.shape).astype(np.float32)
    labels=np.tanh(values[:,0])+.1*values[:,1]
    params=dict(config['training']['booster'],device=str(device),seed=config['seed']+1,
                nthread=config['training']['cpu_threads'] if device.type=='cuda' else 1)
    matrix=xgb.QuantileDMatrix(values,label=labels,base_margin=values[:,-1],max_bin=params['max_bin'],nthread=params['nthread'])
    booster=xgb.Booster(params=params,cache=[matrix])
    times=[]
    for step in range(3):
        started=time.perf_counter();booster.update(matrix,step)
        if device.type=='cuda':torch.cuda.synchronize()
        times.append(time.perf_counter()-started)
    model.trees.import_booster(booster.save_raw(raw_format='json'))
    test_windows,test_clock=preflight_inputs(config['backbone_config'],inference_batch,device)
    with torch.inference_mode():
        test_features,base=model.encoder.group_features(test_windows,test_clock)
        actual=model.forward_group(test_windows,test_clock)[...,0]
    raw=test_features.reshape(-1,test_features.shape[-1]).cpu().numpy()
    expected=booster.predict(xgb.DMatrix(raw,base_margin=base.reshape(-1).cpu().numpy(),nthread=params['nthread']))
    difference=float(np.max(np.abs(actual.cpu().numpy().reshape(-1)-expected)))
    np.testing.assert_allclose(actual.cpu().numpy().reshape(-1),expected,atol=2e-6,rtol=1e-5)
    if actual.dtype!=torch.float32 or not torch.isfinite(actual).all():raise ValueError('便携树输出非有限或非FP32')
    def resume_probe(reload_matrix):
        current_matrix=matrix
        current=xgb.Booster(params=params,cache=[current_matrix])
        for block in (1,2):
            previous=current.save_raw(raw_format='json')
            if block==2 and reload_matrix:
                current_matrix=xgb.QuantileDMatrix(values,label=labels,base_margin=values[:,-1],
                    max_bin=params['max_bin'],nthread=params['nthread'])
            settings=dict(params,seed=config['seed']+block)
            current=xgb.Booster(params=settings,cache=[current_matrix],model_file=previous)
            current.set_param(settings)
            for step in range((block-1)*2,block*2):current.update(current_matrix,step)
        return json.loads(current.save_raw(raw_format='json'))['learner']['gradient_booster']
    resume_check = compare_resume_models(resume_probe(False),resume_probe(True))
    trees=json.loads(booster.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees']
    neural=sum(p.numel() for p in model.encoder.parameters())
    numeric=sum(len(t['split_conditions']) for t in trees)
    return {'status':'passed','scope':'synthetic_neural_tree_device_preflight','device':str(device),
        'train_batch':train_batch,'inference_batch':inference_batch,'batch_unit':'stock_day_feature_encoding',
        'train_target_slots':train_batch*8,'inference_output_slots':int(actual.numel()),
        'parameters':neural+numeric,'frozen_neural_parameters':neural,'synthetic_tree_numeric_parameters':numeric,
        'maximum_tree_numeric_parameters':config['training']['boosting_rounds']*(2**(params['max_depth']+1)-1),
        'tree_training_rows':len(values),'feature_dim':values.shape[1],'feature_encoding_seconds':encoding_seconds,
        'step_seconds':times,'step_timing_scope':'three_native_tree_updates_excludes_feature_encoding_and_quantization',
        'native_portable_max_abs_difference':difference,'xgboost_version':xgb.__version__,
        'block_sampling_resume_tree_model_exact':resume_check['native_tree_document_exact'],
        'block_sampling_resume_prediction_structure_exact':True,'sampling_resume_comparison':resume_check,
        'resume_scope':'four_synthetic_trees_with_matrix_reconstruction_same_device_prediction_parameters_exact_gain_statistics_tolerant',
        'xgboost_build_has_cuda':bool(xgb.build_info()['USE_CUDA']),
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        'gpu_memory_scope':'torch_allocator_only_excludes_native_xgboost','full_dataset_matrix_memory_verified':False,
        'formal_training_performed':False}
