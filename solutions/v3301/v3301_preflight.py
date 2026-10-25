"""只使用合成输入检查冻结编码、活动读出和全时点fast，不运行正式训练。"""
import time
import torch
from v3201_model import MultiFrequencyGRU
from v3201_train import preflight_inputs
from v3301_model import MultiFrequencyClockSpecialists
from v3301_fit import clock_training_step


def run_device_preflight(config, device):
    torch.set_num_threads(config['training']['cpu_threads'] if device.type == 'cuda' else 1)
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA不可用')
        torch.cuda.reset_peak_memory_stats()
    parent = MultiFrequencyGRU(config['backbone_config']).eval()
    model = MultiFrequencyClockSpecialists(config).to(device)
    model.initialize_from_parent(parent.state_dict())
    del parent
    encode_batch = config['training']['feature_inference_batch_size'] if device.type == 'cuda' else 2
    with torch.inference_mode():
        features = model.encoder.group_features(*preflight_inputs(config['backbone_config'], encode_batch, device))
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise FloatingPointError('冻结表示非法')
    count = config['training']['batch_size'] if device.type == 'cuda' else 8
    x = torch.randn(count, config['model']['feature_dim'], device=device)
    targets = torch.randn(count, 2, dtype=torch.float64, device=device)
    weights = torch.ones(count, dtype=torch.float64, device=device)
    head = model.readouts.heads[3]
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['training']['learning_rate'])
    elapsed = []
    for _ in range(3):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        clock_training_step(head, optimizer, x, targets, weights, config, device.type == 'cuda' and torch.cuda.is_bf16_supported())
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - started)
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
            raise FloatingPointError('时点读出梯度不完整')
        if any(p.grad is not None for p in model.encoder.parameters()):
            raise ValueError('冻结编码器出现梯度')
    inference_batch = config['inference']['stock_day_batch_size'] if device.type == 'cuda' else 4
    model.eval()
    with torch.inference_mode():
        values = model.forward_group(*preflight_inputs(config['backbone_config'], inference_batch, device))
    if not torch.isfinite(values).all() or values.dtype != torch.float32:
        raise FloatingPointError('fast推理不满足FP32有限输出')
    return {'status': 'passed', 'scope': 'synthetic_frozen_encoding_clock_training_and_fast',
            'formal_training_performed': False, 'device': str(device),
            'parameters': sum(p.numel() for p in model.parameters()),
            'train_batch': count, 'inference_batch': inference_batch * 8,
            'batch_unit': 'prediction_sample',
            'feature_stock_day_batch': encode_batch, 'active_clock_batch': count,
            'inference_stock_day_batch': inference_batch, 'step_seconds': elapsed,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None}
