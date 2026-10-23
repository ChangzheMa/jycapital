"""完整原始市场上下文与目标股票分批；本模块不导入训练缓存依赖。"""
import hashlib

import torch
import numpy as np

from v2601_grouped import StockDayIndex
from v2601_market import MARKET_KEYS


class ZeroTargets:
    def __getitem__(self, indices):
        return np.zeros(len(indices), np.float64)


class IndexView:
    def __init__(self, base, index):
        self.base, self.index = base, index
        self.eligible = np.ones(len(index.samples), bool)

    def batch(self, groups):
        return self.base.batch(self.index.representatives[groups])


class ResidentIndexView(IndexView):
    def raw_batch(self, groups, destinations=None):
        return self.base.raw_batch(self.index.representatives[groups], destinations)


def index_view(base, indices=None):
    index = StockDayIndex(base, indices)
    return (ResidentIndexView if hasattr(base, 'raw_batch') else IndexView)(base, index)


def native_batches(view, order, targets, start, batch_size, device):
    for offset in range(start, len(order), batch_size):
        selected = order[offset:offset + batch_size]
        windows, clock = view.batch(selected)
        yield offset, selected, {f: torch.from_numpy(x).to(device) for f, x in windows.items()}, \
            torch.from_numpy(clock).to(device), torch.from_numpy(targets[selected]).to(device)


class MarketDayPlan:
    def __init__(self, view):
        self.view = view
        values = view.index.dates.astype('datetime64[D]').astype(np.int64)
        order = np.argsort(values, kind='stable')
        self.days, starts, sizes = np.unique(values[order], return_index=True, return_counts=True)
        self.groups = [np.sort(order[start:start + size]) for start, size in zip(starts, sizes)]
        self.group_days = np.searchsorted(self.days, values)

    def training_order(self, seed, epoch, segment):
        partition = 0 if segment == 'global' else int(segment.replace('-', ''))
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, partition, 2601]))
        date_order = rng.permutation(len(self.days))
        result = []
        for day in date_order:
            groups = self.groups[day]
            targets = groups[self.view.eligible[groups]].copy()
            rng.shuffle(targets)
            result.append(targets)
        return np.concatenate(result) if result else np.empty(0, np.int64)

    def segments(self, order, start, batch_size):
        order = np.asarray(order)
        if order.ndim != 1 or len(np.unique(order)) != len(order) or (order < 0).any() or (order >= len(self.group_days)).any():
            raise ValueError('目标股票日索引必须合法且唯一')
        if not 0 <= start <= len(order):
            raise ValueError('股票日恢复游标越界')
        if not len(order):
            return
        codes = self.group_days[order]
        bounds = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(order)]
        if len(np.unique(codes[bounds[:-1]])) != len(bounds) - 1:
            raise ValueError('同一训练日期必须连续，不能跨日期拼成目标批次')
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if start >= hi:
                continue
            cursor = max(0, start - lo)
            if cursor % batch_size:
                raise ValueError('恢复游标必须对应已完成的日期内目标批次')
            yield int(lo), order[lo:hi], self.groups[int(codes[lo])], int(cursor)

    def record(self, batch_size=None):
        counts = np.array([len(groups) for groups in self.groups], np.int64)
        supervised = np.array([int(self.view.eligible[groups].sum()) for groups in self.groups], np.int64)
        result = {'sampler': 'shuffle_dates_then_eligible_stocks', 'dates': len(self.days),
                'context_stock_days': int(counts.sum()), 'supervised_stock_days': int(supervised.sum()),
                'maximum_context_stocks': int(counts.max()) if len(counts) else 0,
                'dates_sha256': hashlib.sha256(self.days.tobytes()).hexdigest(),
                'group_dates_sha256': hashlib.sha256(self.group_days.tobytes()).hexdigest(),
                'day_context_sizes_sha256': hashlib.sha256(counts.tobytes()).hexdigest(),
                'day_supervised_sizes_sha256': hashlib.sha256(supervised.tobytes()).hexdigest(),
                'context_filter_uses_labels': False, 'learned_context_recomputed_each_update': True}
        if batch_size is not None:
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError('目标股票日批量必须为正整数')
            result['target_stock_day_batch_size'] = batch_size
            result['optimizer_steps_per_full_pass'] = int(((supervised + batch_size - 1) // batch_size).sum())
        return result


def collect_context(model, view, groups, batch_factory, batch_size, device):
    pieces, observed = [], []
    # 使用no_grad而非inference_mode，训练时这些原始观测还要被参数梯度计算保存。
    with torch.no_grad():
        for _, selected, windows, _, _ in batch_factory(view, groups, ZeroTargets(), 0, batch_size, device):
            pieces.append(model.snapshot_features(windows).detach())
            observed.append(np.asarray(selected))
    if not pieces or not np.array_equal(np.concatenate(observed), groups):
        raise ValueError('市场原始上下文与成员顺序不一致')
    snapshots = torch.cat(pieces, dim=0)
    membership = torch.as_tensor(view.index.samples[groups] >= 0, dtype=torch.bool, device=device)
    return snapshots, membership


def context_rows(groups, selected, device):
    positions = np.searchsorted(groups, selected)
    if (positions >= len(groups)).any() or not np.array_equal(groups[positions], selected):
        raise ValueError('目标股票不在完整当日上下文中')
    return torch.as_tensor(positions, dtype=torch.long, device=device)


def market_training_batches(model, view, order, targets, start, batch_size, device, config, batch_factory):
    plan = view.market_plan
    context_batch = config['training']['market_conditioning']['context_stock_day_batch_size']
    for offset, target_order, groups, cursor in plan.segments(order, start, batch_size):
        snapshots, membership = collect_context(model, view, groups, batch_factory, context_batch, device)
        for local, selected, windows, clock, target in batch_factory(view, target_order, targets, cursor, batch_size, device):
            augmented = {**windows, MARKET_KEYS[0]: snapshots, MARKET_KEYS[1]: membership,
                         MARKET_KEYS[2]: context_rows(groups, selected, device)}
            yield offset + local, selected, augmented, clock, target


def predict_groups(model, view, device, config, batch_factory):
    if model.training:
        raise ValueError('市场共享推理必须使用eval模式')
    plan = MarketDayPlan(view)
    result = np.empty((len(view.index.samples), 8), np.float64)
    context_batch = config['training']['market_conditioning']['context_stock_day_batch_size']
    target_batch = config['inference']['stock_day_batch_size']
    with torch.inference_mode():
        for groups in plan.groups:
            snapshots, membership = collect_context(model, view, groups, batch_factory, context_batch, device)
            state = model.prepare_market(snapshots, membership)
            for _, selected, windows, clock, _ in batch_factory(view, groups, ZeroTargets(), 0, target_batch, device):
                scores = model.forward_group(windows, clock, market_state=state,
                    target_rows=context_rows(groups, selected, device))[..., 0]
                result[selected] = (scores / config['inference']['factor_scale']).cpu().numpy()
    return result
