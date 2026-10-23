"""本地市场上下文传输与训练诊断；不作为平台数值入口依赖。"""
import torch

from v2601_resident import ResidentBatches
from v2601_context import native_batches


def plain_batches(view, order, targets, start, batch_size, device):
    if hasattr(view, 'raw_batch'):
        yield from ResidentBatches(view, order, targets, start, batch_size, device)
    else:
        yield from native_batches(view, order, targets, start, batch_size, device)


def centered_rms(values, membership):
    values = values.detach().float()
    weights = membership.float()
    count = weights.sum(dim=0, keepdim=True).clamp_min(1)
    mean = (values * weights).sum(dim=0, keepdim=True) / count
    return (((values-mean).square() * weights).sum() / weights.sum().clamp_min(1)).sqrt()


DIAGNOSTICS = ('raw_main_loss', 'date_multiplier', 'market_correction_rms', 'prediction_rms',
               'modulation_scale_rms', 'modulation_shift_rms')


def diagnostic_progress():
    return {key + '_sum': 0. for key in DIAGNOSTICS}


def accumulate_diagnostics(state, step):
    for key in DIAGNOSTICS:
        state[key + '_sum'] += step[key] * step['rows']


def diagnostic_history(state):
    return {'train_' + key: state[key + '_sum'] / state['rows'] for key in DIAGNOSTICS}
