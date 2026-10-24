"""仅训练区间的分阶段表征学习；推理模块不导入本文件。"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from artifacts import log_event, now, read_json, sha256, write_json
from v3101_grouped import StockDayIndex
from v3101_model import DAILY_BARS


def validate_pretraining(config):
    spec = config['training'].get('pretraining', {})
    expected = {'method': 'clock_centered_masked_vicreg', 'explicit_clock_input': False,
                'selection': 'fixed_epochs_no_holdout',
                'sampling': 'all_training_stock_days_independent_of_labels'}
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('预训练必须按固定轮数、无标签股票日及无显式时点输入执行')
    for key in ('epochs', 'stock_day_batch_size', 'projector_hidden_dim', 'projector_output_dim'):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError('预训练整数配置非法: ' + key)
    if spec['stock_day_batch_size'] < 2:
        raise ValueError('预训练batch至少需要两个股票日')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip', 'invariance_weight',
                'variance_weight', 'covariance_weight', 'variance_target', 'variance_epsilon'):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('预训练正数配置非法: ' + key)
    if not 0 < spec.get('mask_probability', 0) < 1:
        raise ValueError('遮挡概率须在(0,1)内')


def make_projector(config):
    spec = config['training']['pretraining']
    return nn.Sequential(nn.Linear(config['model']['fusion_hidden_dim'], spec['projector_hidden_dim']),
                         nn.GELU(), nn.Linear(spec['projector_hidden_dim'], spec['projector_output_dim']))


def mask_windows(windows, probability, *, keep=None):
    """相同日期／半小时跨频同步遮挡，返回新张量；不裁剪或拼接未来。"""
    first = next(iter(windows.values()))
    batch = first.shape[0]
    days = max(x.shape[-1] // DAILY_BARS[f] for f, x in windows.items())
    if keep is None:
        keep = torch.rand(batch, days, 8, device=first.device) >= probability
    if keep.dtype != torch.bool or tuple(keep.shape) != (batch, days, 8):
        raise ValueError('跨频遮挡网格形状或类型不符')
    result = {}
    for frequency, value in windows.items():
        bars = DAILY_BARS[frequency]
        if value.shape[-1] % bars:
            raise ValueError('遮挡要求固定完整交易日网格')
        own_days = value.shape[-1] // bars
        mask = keep[:, -own_days:].repeat_interleave(bars // 8, dim=-1).reshape(batch, 1, -1)
        result[frequency] = value.masked_fill(~mask, 0)
    return result


def representation_loss(left, right, present, spec):
    """同一时点内计方差、去时点均值后合并协方差；不把时点当独立证据。"""
    if left.shape != right.shape or left.ndim != 3 or left.shape[:2] != present.shape:
        raise ValueError('表征或请求掩码形状不一致')
    if left.shape[1] != 8 or present.dtype != torch.bool:
        raise ValueError('预训练要求八时点布尔请求掩码')
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError('表征含非有限数')
    # 原始主网络可以BF16投影；损失与协方差固定FP32，手算测试允许FP64。
    dtype = torch.float64 if left.dtype == torch.float64 else torch.float32
    left, right = left.to(dtype), right.to(dtype)
    weight = present.to(dtype)
    count = weight.sum(dim=0)
    usable = count >= 2
    if not usable.any():
        raise ValueError('当前batch没有至少两个有效股票日的时点')
    width = left.shape[-1]
    invariance = ((left - right).square() * weight[..., None]).sum() / (weight.sum() * width)
    denominator = (count[usable] - 1).sum()
    off_diagonal = ~torch.eye(width, dtype=torch.bool, device=left.device)
    variances, covariances, std_means, active_dimensions = [], [], [], []
    for values in (left, right):
        mean = (values * weight[..., None]).sum(dim=0) / count.clamp_min(1)[..., None]
        centered = (values - mean) * weight[..., None] * usable[None, :, None]
        variance = centered.square().sum(dim=0) / (count - 1).clamp_min(1)[..., None]
        std = torch.sqrt(variance[usable] + spec['variance_epsilon'])
        variances.append(F.relu(spec['variance_target'] - std).mean())
        covariance = torch.einsum('bqd,bqe->de', centered, centered) / denominator
        covariances.append(covariance[off_diagonal].square().sum() / width)
        std_means.append(std.mean())
        active_dimensions.append((std > 0.1 * spec['variance_target']).to(dtype).mean())
    variance = (variances[0] + variances[1]) / 2
    covariance = covariances[0] + covariances[1]
    loss = (spec['invariance_weight'] * invariance + spec['variance_weight'] * variance +
            spec['covariance_weight'] * covariance)
    details = {'invariance': invariance.detach(), 'variance': variance.detach(),
               'covariance': covariance.detach(), 'std_mean': torch.stack(std_means).mean().detach(),
               'dimensions_above_0_1_fraction': torch.stack(active_dimensions).mean().detach(),
               'valid_clocks': usable.sum().detach(), 'present_queries': weight.sum().detach()}
    return loss, details


def pretraining_step(model, projector, optimizer, windows, clock, present, config, amp):
    model.train()
    projector.train()
    optimizer.zero_grad(set_to_none=True)
    spec = config['training']['pretraining']
    first = mask_windows(windows, spec['mask_probability'])
    second = mask_windows(windows, spec['mask_probability'])
    with torch.autocast(clock.device.type, dtype=torch.bfloat16, enabled=amp):
        left = projector(model.forward_group_features(first, clock))
        right = projector(model.forward_group_features(second, clock))
    loss, details = representation_loss(left, right, present.bool(), spec)
    loss.backward()
    nn.utils.clip_grad_norm_([*model.parameters(), *projector.parameters()], spec['gradient_clip'],
                            error_if_nonfinite=True)
    optimizer.step()
    return {'loss': float(loss.detach()), **{k: float(v) for k, v in details.items()}}


class UnlabelledView:
    def __init__(self, base):
        self.base = base
        self.index = StockDayIndex(base)
        self.targets = self.index.samples >= 0  # 仅请求存在性，绝不读取收益或有效标签掩码。

    def batch(self, indices):
        return self.base.batch(self.index.representatives[indices])


class ResidentUnlabelledView(UnlabelledView):
    def raw_batch(self, indices, destinations=None):
        return self.base.raw_batch(self.index.representatives[indices], destinations)


def unlabelled_view(block):
    return (ResidentUnlabelledView if hasattr(block, 'raw_batch') else UnlabelledView)(block)


def shuffled_pretraining_groups(view, seed, epoch, segment):
    order = np.arange(len(view.index.samples), dtype=np.int64)
    partition = 0 if segment == 'global' else int(segment.replace('-', ''))
    np.random.default_rng(np.random.SeedSequence([seed, 3101, epoch, partition])).shuffle(order)
    return order


def record_view(run, segment, view):
    path = run / 'pretraining-data.json'
    identity = {'format': 'jycapital-unlabelled-stock-days-v1',
                'data_manifest_sha256': sha256(run / 'data-manifest.json'), 'segments': {}}
    record = {'groups': len(view.index.samples), 'membership_sha256': view.index.digest(),
              'sample_rows': view.index.sample_count, 'labels_read_by_pretext': False}
    if path.exists():
        saved = read_json(path)
        if saved['format'] != identity['format'] or saved['data_manifest_sha256'] != identity['data_manifest_sha256']:
            raise ValueError('预训练数据身份已改变')
        identity = saved
        if segment in identity['segments'] and identity['segments'][segment] != record:
            raise ValueError('预训练股票日成员已改变')
    identity['segments'][segment] = record
    write_json(path, identity)


def verified_pretraining(run, config):
    record = read_json(run / 'pretraining.json')
    if (record['status'] != 'complete' or record['config'] != config['training']['pretraining'] or
            record['data_manifest_sha256'] != sha256(run / 'data-manifest.json') or
            record['model_config_sha256'] != sha256(run / 'config.json') or
            record['weights_sha256'] != sha256(run / 'pretrained.pt') or
            record['membership_sha256'] != sha256(run / 'pretraining-data.json')):
        raise ValueError('已完成预训练的配置、数据或权重身份不一致')
    return record


def run_pretraining(store, run, config, manifest, engine, resume=False):
    """固定阶段只加载train；完整凭据存在时只核验，不重新做预训练。"""
    run = Path(run)
    validate_pretraining(config)
    if (run / 'pretraining.json').exists():
        return verified_pretraining(run, config)
    if (run / 'last.pt').exists():
        raise ValueError('已有收益训练checkpoint但缺少完成的预训练凭据')
    device = engine.configure(config)
    spec = config['training']['pretraining']
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    model = engine.MultiFrequencyGRU(config).to(device)
    projector = make_projector(config).to(device)
    optimizer = torch.optim.AdamW([*model.parameters(), *projector.parameters()],
                                 lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    data_digest = sha256(run / 'data-manifest.json')
    state = {'epoch': 1, 'month_cursor': 0, 'group_cursor': 0, 'optimizer_steps': 0,
             'epoch_steps': 0, 'groups': 0, 'sums': {}, 'seconds': 0.0, 'history': [],
             'skipped_singleton_groups': 0}
    checkpoint_path = run / 'pretraining-last.pt'
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['data_manifest_sha256'] != data_digest:
            raise ValueError('预训练恢复的配置或数据不一致')
        model.load_state_dict(checkpoint['state_dict'])
        projector.load_state_dict(checkpoint['projector'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        state = checkpoint['progress']
        engine.restore_rng(checkpoint, device)
    manifest.update(stage='representation_pretraining')
    write_json(run / 'manifest.json', manifest)
    log_event(run, 'pretraining_started', epochs=spec['epochs'], resume=resume,
              optimizer_steps=state['optimizer_steps'], selection='fixed_epochs_no_holdout')
    checkpoint_clock = time.perf_counter()

    def save():
        nonlocal checkpoint_clock
        current = time.perf_counter()
        state['seconds'] += current - checkpoint_clock
        checkpoint_clock = current
        engine.atomic_torch_save({'state_dict': model.state_dict(), 'projector': projector.state_dict(),
            'optimizer': optimizer.state_dict(), 'config': config, 'progress': state,
            'data_manifest_sha256': data_digest, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, checkpoint_path)
        write_json(run / 'pretraining-progress.json', {k: state[k] for k in
                   ('epoch', 'month_cursor', 'group_cursor', 'optimizer_steps')})

    save()
    for epoch in range(state['epoch'], spec['epochs'] + 1):
        months = np.random.default_rng(config['seed'] + 3101 + epoch).permutation(store.months('train')).tolist()
        for month, block in engine.blocks(store, 'train', months[state['month_cursor']:]):
            view = unlabelled_view(block)
            record_view(run, month, view)
            order = shuffled_pretraining_groups(view, config['seed'], epoch, month)
            for offset, indices, windows, clock, present in engine.training_batches(
                    view, order, view.targets, state['group_cursor'], spec['stock_day_batch_size'], device):
                if len(indices) < 2 or not (present.sum(dim=0) >= 2).any():
                    state['skipped_singleton_groups'] += len(indices)
                else:
                    step = pretraining_step(model, projector, optimizer, windows, clock, present, config, amp)
                    for key, value in step.items():
                        state['sums'][key] = state['sums'].get(key, 0.0) + value * len(indices)
                    state['groups'] += len(indices)
                    state['optimizer_steps'] += 1
                    state['epoch_steps'] += 1
                state['group_cursor'] = offset + len(indices)
                if state['epoch_steps'] % config['training']['checkpoint_batches'] == 0:
                    save()
            state.update(month_cursor=state['month_cursor'] + 1, group_cursor=0)
            save()
            del view, block
        if not state['groups']:
            raise ValueError('预训练整轮无有效股票日batch')
        row = {'epoch': epoch, **{k: v / state['groups'] for k, v in state['sums'].items()},
               'groups': state['groups'], 'optimizer_steps': state['epoch_steps'],
               'skipped_singleton_groups': state['skipped_singleton_groups'],
               'seconds': state['seconds'] + time.perf_counter() - checkpoint_clock}
        state['history'].append(row)
        log_event(run, 'pretraining_epoch_finished', **row)
        state.update(epoch=epoch + 1, month_cursor=0, group_cursor=0, epoch_steps=0,
                     groups=0, sums={}, seconds=0.0, skipped_singleton_groups=0)
        checkpoint_clock = time.perf_counter()
        save()
    engine.atomic_torch_save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                             'config': config, 'data_manifest_sha256': data_digest,
                             'epochs_completed': spec['epochs']}, run / 'pretrained.pt')
    record = {'status': 'complete', 'completed_at': now(), 'config': spec,
              'data_manifest_sha256': data_digest, 'model_config_sha256': sha256(run / 'config.json'),
              'weights_sha256': sha256(run / 'pretrained.pt'),
              'membership_sha256': sha256(run / 'pretraining-data.json'),
              'epochs_completed': len(state['history']), 'optimizer_steps': state['optimizer_steps'],
              'history': state['history'], 'holdout_used': False, 'return_labels_used': False,
              'inference_parameters': sum(p.numel() for p in model.parameters()),
              'pretraining_only_parameters': sum(p.numel() for p in projector.parameters()),
              'supervised_optimizer_and_ema': 'fresh_after_fixed_pretraining'}
    write_json(run / 'pretraining.json', record)
    log_event(run, 'pretraining_completed', optimizer_steps=state['optimizer_steps'],
              weights_sha256=record['weights_sha256'])
    return record
