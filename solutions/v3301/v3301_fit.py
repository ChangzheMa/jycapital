"""八时点串行拟合和独立验证选择，不接触测试缓存。"""
import copy
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from artifacts import read_json, sha256, now
from early_stopping import select_epoch, should_stop, patience_epochs
from baseline_evaluation import evaluate
from v3301_checkpoint import atomic_json, atomic_save, rng_state, restore_rng, cpu_state
from v3301_readout import CLOCKS
from v3301_objective import clock_objective
from v3301_text_artifact import state_digest


def update_ema(ema, head, count, epoch, config):
    averaging = config['training']['weight_averaging']
    if epoch <= averaging['warmup_epochs']:
        ema.load_state_dict(head.state_dict())
        return
    decay = averaging['decay'] ** (count / config['training']['ema_reference_samples'])
    with torch.no_grad():
        for key, value in ema.state_dict().items():
            # 相同参数的差值为零，避免无优化更新时仅因两次舍入改变EMA。
            value.lerp_(head.state_dict()[key].float(), 1 - decay)


def clock_training_step(head, optimizer, features, targets, weights, config, amp):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(features.device.type, dtype=torch.bfloat16, enabled=amp):
        prediction, experts, logits = head(features)
    loss, details = clock_objective(prediction, experts, logits, targets, weights, config['training'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), config['training']['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    return {'loss': float(loss.detach()), **details}


def evaluate_head(head, cache, clock, config, protocol, device):
    rows = cache.clock_rows(clock)
    if not len(rows):
        raise ValueError('验证缺少固定预测时点')
    head.eval()
    values = np.empty(len(rows), np.float64)
    batch = config['training']['clock_evaluation_batch_size']
    with torch.inference_mode():
        for start in range(0, len(rows), batch):
            x = torch.from_numpy(np.array(cache.arrays['features'][rows[start:start + batch]], copy=True)).to(device)
            values[start:start + len(x)] = head(x)[0][:, 0].cpu().numpy()
    keys = cache.keys.iloc[rows].reset_index(drop=True)
    predictions = keys[['date', 'instrument']].copy()
    predictions['factor'] = values
    result = evaluate(predictions, keys, protocol)
    target = np.asarray(cache.arrays['targets'][rows, 0])
    valid = np.isfinite(target)
    difference = np.abs(values[valid] - target[valid])
    delta = config['training']['huber_delta']
    loss = float(np.where(difference <= delta, .5 * difference ** 2, delta * (difference - .5 * delta)).mean())
    return predictions, result, loss


def eligible(metrics):
    return (metrics['coverage'] == 1. and metrics['min_cross_section_coverage'] == 1.
            and metrics['ic_undefined_cross_sections'] == 0 and metrics['rank_ic_undefined_cross_sections'] == 0
            and metrics['ic_mean'] is not None and np.isfinite(metrics['ic_mean']))


def save_evaluation(run, split, predictions, result):
    directory = Path(run) / 'evaluation' / split
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(directory / 'predictions.parquet', index=False)
    atomic_json(directory / 'metrics.json', result[0])
    result[1].to_parquet(directory / 'cross-sections.parquet', index=False)
    result[2].to_csv(directory / 'monthly.csv', index=False, encoding='utf-8-sig')


def evaluate_bank(bank, cache, config, protocol, device):
    pieces = []
    for clock, head in enumerate(bank.heads):
        head.to(device)
        predictions, _, _ = evaluate_head(head, cache, clock, config, protocol, device)
        pieces.append(predictions)
        head.cpu()
    frame = pd.concat(pieces, ignore_index=True).sort_values(['date', 'instrument']).reset_index(drop=True)
    return frame, evaluate(frame, cache.keys, protocol)


def fit_readouts(model, train, validation, config, protocol, run, device, resume=False, report=None):
    run = Path(run)
    spec = config['training']
    binding = {'parent_model_sha256': config['model']['parent_model_sha256'],
               'train_features': sha256(train.directory / 'manifest.json'),
               'validation_features': sha256(validation.directory / 'manifest.json'),
               'encoder': state_digest(model.encoder.state_dict())}
    state = {'clock': 0, 'active': None, 'completed': [], 'history': [], 'optimizer_steps': 0}
    checkpoint = None
    if resume and (run / 'last.pt').exists():
        checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['binding'] != binding:
            raise ValueError('恢复时父模型、特征或配置改变')
        model.readouts.load_state_dict(checkpoint['readouts_state'])
        state = checkpoint['progress']
        restore_rng(checkpoint['rng'], device)
        for item in state['completed']:
            path = run / 'clocks' / str(item['clock_id']) / 'best.pt'
            if sha256(path) != item['checkpoint_sha256'] or state_digest(model.readouts.heads[item['clock_id']].state_dict()) != item['tensor_sha256']:
                raise ValueError('已完成时点的选定参数改变')
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    for clock in range(state['clock'], 8):
        directory = run / 'clocks' / str(clock)
        directory.mkdir(parents=True, exist_ok=True)
        head = model.readouts.heads[clock].to(device)
        ema = copy.deepcopy(head).eval()
        optimizer = torch.optim.AdamW(head.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
        epoch_clock = time.perf_counter()
        if state['active'] is None:
            _, initial_result, initial_loss = evaluate_head(head, validation, clock, config, protocol, device)
            if not eligible(initial_result[0]):
                raise ValueError('父读出未通过该时点完整验证')
            progress = {'epoch': 1, 'cursor': 0, 'best': initial_result[0]['ic_mean'], 'best_epoch': 0,
                        'stale': 0, 'optimizer_steps': 0, 'ema_updates': 0, 'loss_sums': {}, 'rows': 0,
                        'epoch_steps': 0, 'epoch_seconds': 0., 'initial_metric': initial_result[0]['ic_mean']}
            state['active'] = progress
            chosen_head = {'state_dict': cpu_state(head), 'clock_id': clock, 'epoch': 0,
                'validation_ic': progress['best'], 'ema_updates': 0, 'weight_source': 'parent_ema', 'binding': binding}
            atomic_json(directory / 'initial-validation.json', {'metrics': initial_result[0], 'huber_loss': initial_loss})
        else:
            progress = state['active']
            optimizer.load_state_dict(checkpoint['optimizer'])
            ema.load_state_dict(checkpoint['ema_state'])
            restore_rng(checkpoint['rng'], device)
            if progress['optimizer_steps'] != progress['ema_updates']:
                raise ValueError('时点EMA与优化器更新数不一致')
            chosen_head = checkpoint['active_best']
            if chosen_head['epoch'] != progress['best_epoch'] or chosen_head['binding'] != binding:
                raise ValueError('恢复时当前时点最佳权重记录不同')

        def save_active():
            nonlocal epoch_clock
            current = time.perf_counter()
            progress['epoch_seconds'] += current - epoch_clock
            epoch_clock = current
            payload = {'config': config, 'binding': binding, 'progress': copy.deepcopy(state),
                       'readouts_state': cpu_state(model.readouts), 'optimizer': optimizer.state_dict(),
                       'ema_state': cpu_state(ema), 'rng': rng_state(device),
                       'active_best': chosen_head}
            atomic_save(run / 'last.pt', payload)
            atomic_json(run / 'progress.json', {'stage': 'clock_training', 'clock_id': clock, 'clock': CLOCKS[clock],
                'epoch': progress['epoch'], 'sample_cursor': progress['cursor'], 'optimizer_steps': state['optimizer_steps'],
                'completed_clocks': len(state['completed'])})
            if report:
                report('checkpoint_saved', checkpoint='last.pt', clock_id=clock, epoch=progress['epoch'], sample_cursor=progress['cursor'])

        save_active()
        rows = train.clock_rows(clock, finite_targets=True)
        if not len(rows):
            raise ValueError('时点没有有效训练样本')
        for epoch in range(progress['epoch'], spec['epochs'] + 1):
            if should_stop(progress):
                break
            order = rows.copy()
            np.random.default_rng(np.random.SeedSequence([config['seed'], clock, epoch])).shuffle(order)
            for start in range(progress['cursor'], len(order), spec['batch_size']):
                selected = order[start:start + spec['batch_size']]
                features = torch.from_numpy(np.array(train.arrays['features'][selected], copy=True)).to(device)
                targets = torch.from_numpy(np.array(train.arrays['targets'][selected], copy=True)).to(device)
                weights = torch.from_numpy(np.array(train.arrays['weights'][selected], copy=True)).to(device)
                step = clock_training_step(head, optimizer, features, targets, weights, config, amp)
                update_ema(ema, head, len(selected), epoch, config)
                progress['optimizer_steps'] += 1
                progress['ema_updates'] += 1
                state['optimizer_steps'] += 1
                progress['epoch_steps'] += 1
                progress['rows'] += len(selected)
                for key, value in step.items():
                    progress['loss_sums'][key] = progress['loss_sums'].get(key, 0.) + value * len(selected)
                progress['cursor'] = start + len(selected)
                if progress['epoch_steps'] % spec['checkpoint_batches'] == 0:
                    save_active()
            started = time.perf_counter()
            _, result, validation_loss = evaluate_head(ema, validation, clock, config, protocol, device)
            validation_seconds = time.perf_counter() - started
            improved = select_epoch(progress, result[0]['ic_mean'], epoch, eligible(result[0]))
            if improved:
                chosen_head = {'state_dict': cpu_state(ema), 'clock_id': clock, 'epoch': epoch,
                    'validation_ic': progress['best'], 'ema_updates': progress['ema_updates'], 'weight_source': 'clock_ema', 'binding': binding}
            row = {'clock_id': clock, 'clock': CLOCKS[clock], 'epoch': epoch,
                   'train_loss': progress['loss_sums']['loss'] / progress['rows'],
                   'train_huber_loss': progress['loss_sums']['main_loss'] / progress['rows'],
                   'validation_loss': validation_loss, 'validation_ic_mean': result[0]['ic_mean'],
                   'validation_rank_ic_mean': result[0]['rank_ic_mean'], 'best_epoch': progress['best_epoch'],
                   'improved': bool(improved), 'stale_epochs': progress['stale'],
                   'selection_eligible': bool(eligible(result[0])),
                   'patience_epochs': patience_epochs(progress['best_epoch']), 'train_rows': progress['rows'],
                   'optimizer_steps': progress['epoch_steps'], 'ema_updates': progress['ema_updates'],
                   'validation_seconds': validation_seconds, 'seconds': progress['epoch_seconds'] + time.perf_counter() - epoch_clock}
            state['history'].append(row)
            pd.DataFrame(state['history']).to_csv(run / 'history.csv', index=False, encoding='utf-8-sig')
            if report:
                report('clock_epoch_finished', **row)
            progress.update(epoch=epoch + 1, cursor=0, loss_sums={}, rows=0, epoch_steps=0, epoch_seconds=0.)
            epoch_clock = time.perf_counter()
            save_active()
        best = chosen_head
        atomic_save(directory / 'best.pt', best)
        if best['epoch'] != progress['best_epoch'] or best['binding'] != binding or best['clock_id'] != clock:
            raise ValueError('时点选模记录与最佳参数不一致')
        # 保留各阶段末态以复核优化器／EMA；完成后不会再训练这个时点。
        atomic_save(directory / 'last.pt', torch.load(run / 'last.pt', map_location='cpu', weights_only=True))
        head.load_state_dict(best['state_dict'])
        head.zero_grad(set_to_none=True)
        head.cpu()
        selection = {'clock_id': clock, 'clock': CLOCKS[clock], 'best_epoch': best['epoch'],
                     'kept_parent': best['epoch'] == 0, 'best_validation_ic': best['validation_ic'],
                     'parent_validation_ic': progress['initial_metric'], 'selected_ema_updates': best['ema_updates'],
                     'epochs_completed': progress['epoch'] - 1, 'optimizer_steps': progress['optimizer_steps'],
                     'stop_reason': 'early_stopping' if should_stop(progress) else 'max_epochs',
                     'patience_epochs': patience_epochs(progress['best_epoch']),
                     'checkpoint_sha256': sha256(directory / 'best.pt'), 'tensor_sha256': state_digest(head.state_dict())}
        state['completed'].append(selection)
        atomic_json(directory / 'selection.json', selection)
        state.update(clock=clock + 1, active=None)
        checkpoint = {'config': config, 'binding': binding, 'progress': copy.deepcopy(state),
                      'readouts_state': cpu_state(model.readouts), 'rng': rng_state(device)}
        atomic_save(run / 'last.pt', checkpoint)
        if report:
            report('clock_completed', **selection)
        del optimizer, ema
    if len(state['completed']) != 8 or state_digest(model.encoder.state_dict()) != binding['encoder']:
        raise ValueError('八时点选择未完成或冻结编码器改变')
    frame, result = evaluate_bank(model.readouts, validation, config, protocol, device)
    if not eligible(result[0]):
        raise ValueError('组合验证覆盖或相关性检查失败')
    save_evaluation(run, 'validation', frame, result)
    checkpoint = {'state_dict': cpu_state(model), 'config': config, 'protocol': protocol,
                  'weight_source': 'per_clock_validation_ema_or_parent', 'clock_selections': state['completed']}
    atomic_save(run / 'best.pt', checkpoint)
    selection = {'selection_kind': 'independent_clock_validation', 'best_epoch': None,
        'best_validation_ic': result[0]['ic_mean'], 'best_validation_metric': result[0]['ic_mean'],
        'selection_metric': 'ic_mean', 'epochs_completed': len(state['history']), 'epoch_count_unit': 'clock_epoch',
        'selected_epochs_by_clock': {item['clock']: item['best_epoch'] for item in state['completed']},
        'clock_selections': state['completed'], 'optimizer_steps': state['optimizer_steps'],
        'model_sha256': sha256(run / 'best.pt'), 'selected_at': now(),
        'evaluation_weights': 'per_clock_ema_or_parent', 'stop_reason': 'all_clock_stages_complete'}
    atomic_json(run / 'selection.json', selection)
    return selection
