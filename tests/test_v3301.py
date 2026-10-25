"""独立时点选模、恢复和真实标签缓存的CPU实现验证。"""
import copy
import importlib
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solutions/v3301'))
import v3301_train as trainer
import v3301_fit as fit
from v3301_model import MultiFrequencyClockSpecialists
from v3301_cache import build_features, FeatureCache, identity_for
from v3301_text_artifact import state_digest
from v3201_model import MultiFrequencyGRU
from v3201_targets import prepare_rank_training_targets
from test_v3201 import small_config as parent_small, Block, inputs

CONFIG = json.loads((ROOT / 'solutions/v3301/model-config.json').read_text(encoding='utf-8'))
PROTOCOL = json.loads((ROOT / 'evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))
PROTOCOL['min_cross_section_size'] = 3
torch.set_num_threads(1)


def small_config():
    c = copy.deepcopy(CONFIG)
    c['backbone_config'] = parent_small(resident=True)
    c['data'] = copy.deepcopy(c['backbone_config']['data'])
    c['model']['feature_dim'] = 16
    c['training'].update(epochs=2, batch_size=4, cpu_threads=1, checkpoint_batches=1,
                         feature_checkpoint_batches=1, feature_inference_batch_size=2, clock_evaluation_batch_size=8)
    c['inference'].update(stock_day_batch_size=2, batch_size=16)
    return c


def model_from_parent(config):
    torch.manual_seed(42)
    parent = MultiFrequencyGRU(config['backbone_config']).eval()
    model = MultiFrequencyClockSpecialists(config)
    model.initialize_from_parent(parent.state_dict())
    return model, parent


class ArrayCache:
    def __init__(self, directory, split, width):
        self.directory = directory
        directory.mkdir()
        (directory / 'manifest.json').write_text(json.dumps({'synthetic': True, 'split': split}), encoding='utf-8')
        first = '2020-01-02' if split == 'train' else '2024-01-02'
        keys = pd.DataFrame([(pd.Timestamp(first) + pd.Timedelta(days=d, minutes=int(t[:2])*60+int(t[3:])), f'S{s}')
                            for d in range(2) for t in CONFIG['data']['prediction_clock']['times'] for s in range(6)], columns=['date', 'instrument'])
        rng = np.random.default_rng(11 if split == 'train' else 12)
        keys['vwap_return'] = rng.normal(0, .001, len(keys))
        self.keys = keys
        targets, _ = trainer.parent_train.prepare_training_targets(keys, PROTOCOL)
        ranks, _ = prepare_rank_training_targets(keys, PROTOCOL)
        self.arrays = {'features': rng.normal(size=(len(keys), width)).astype(np.float32),
                       'targets': np.stack([targets, ranks], -1), 'weights': np.linspace(.5, 1, len(keys)),
                       'clock': np.tile(np.repeat(np.arange(8), 6), 2).astype(np.uint8)}

    def clock_rows(self, clock, finite_targets=False):
        selected = self.arrays['clock'] == clock
        if finite_targets:
            selected &= np.isfinite(self.arrays['targets']).all(1)
        return np.flatnonzero(selected)


class ClockTests(unittest.TestCase):
    def test_bfloat16_clock_update_keeps_encoder_frozen_and_gradients_finite(self):
        c = small_config(); model, _ = model_from_parent(c)
        before = state_digest(model.encoder.state_dict())
        head = model.readouts.heads[4]
        optimizer = torch.optim.AdamW(head.parameters(), lr=.0001)
        result = fit.clock_training_step(head, optimizer, torch.randn(8, 16),
            torch.randn(8, 2, dtype=torch.float64), torch.ones(8, dtype=torch.float64), c, True)
        self.assertTrue(np.isfinite(result['loss']))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertEqual(before, state_digest(model.encoder.state_dict()))

    def test_clock_objective_matches_parent_when_only_one_clock_has_targets(self):
        from v3201_objective import combined_objective
        from v3301_objective import clock_objective
        c = small_config()
        prediction = torch.randn(6, 2, requires_grad=True)
        experts = torch.randn(6, 4, 2, requires_grad=True)
        logits = torch.randn(6, 4, requires_grad=True)
        target, weights = torch.randn(6, 2, dtype=torch.float64), torch.rand(6, dtype=torch.float64) + .5
        actual, _ = clock_objective(prediction, experts, logits, target, weights, c['training'])
        padded = torch.full((6, 8, 2), float('nan'), dtype=torch.float64)
        padded[:, 2] = target
        expected, _ = combined_objective(prediction[:, None].expand(-1, 8, -1), experts[:, None].expand(-1, 8, -1, -1),
                                         logits[:, None].expand(-1, 8, -1), padded, weights, c['training'])
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
        left = torch.autograd.grad(actual, (prediction, experts, logits), retain_graph=True)
        right = torch.autograd.grad(expected, (prediction, experts, logits))
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-7)

    def test_feature_metadata_initialization_can_resume(self):
        import v3301_cache as cache_module
        c = small_config(); model, _ = model_from_parent(c)
        block = Block('2024-01', 3)
        _, identity = identity_for(c, {'synthetic': True}, {'cache_key': 'synthetic'}, 'validation', state_digest(model.encoder.state_dict()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'features'
            with patch.object(cache_module, 'prepare_rank_training_targets', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                build_features(path, identity, block, block.keys, model.encoder, c, PROTOCOL, 'validation')
            self.assertEqual(json.loads((path.with_name('features.building') / 'progress.json').read_text(encoding='utf-8'))['status'], 'initializing')
            build_features(path, identity, block, block.keys, model.encoder, c, PROTOCOL, 'validation')
            result = FeatureCache(path, identity)
            result.close()

    def test_parent_initialization_features_and_fast_causality(self):
        c = small_config()
        m, parent = model_from_parent(c)
        m.eval()
        windows, latest = inputs(c['backbone_config'], 2)
        captured = []
        hook = parent.shared.register_forward_hook(lambda module, args, value: captured.append(value.detach()))
        with torch.no_grad():
            expected = parent.forward_group(windows, latest)
            actual = m.forward_group(windows, latest)
            features = m.encoder.group_features(windows, latest)
        hook.remove()
        torch.testing.assert_close(features, captured[0].reshape(8, 2, -1).transpose(0, 1), rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)
        for q in range(8):
            clock = torch.nn.functional.one_hot(torch.full((2,), q), 8).float()
            altered = {f: x.clone() for f, x in windows.items()}
            for f, x in altered.items():
                bars = 240 // int(f[3:-1])
                cutoff = x.shape[-1] - bars + q * bars // 8
                x[:, :, cutoff:] = 1000
            torch.testing.assert_close(m.encoder.group_features(altered, latest)[:, q], features[:, q], rtol=0, atol=0)
            torch.testing.assert_close(m.forward_heads(windows, clock), actual[:, q], rtol=1e-5, atol=2e-7)
        self.assertTrue(all(p.grad is None for p in m.encoder.parameters()))
        self.assertEqual(sum(p.numel() for p in MultiFrequencyClockSpecialists(CONFIG).parameters()), 3913772)

    def test_cache_resume_targets_not_placeholder_and_corruption_rejected(self):
        c = small_config()
        model, _ = model_from_parent(c)
        block = Block('2024-01', 3)
        block.targets = np.zeros(len(block.keys))
        key, identity = identity_for(c, {'synthetic': True}, {'cache_key': 'synthetic'}, 'validation', state_digest(model.encoder.state_dict()))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def interrupt(event, **fields):
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                build_features(root / 'resumed', identity, block, block.keys, model.encoder, c, PROTOCOL, 'validation', interrupt)
            p = json.loads((root / 'resumed.building/progress.json').read_text(encoding='utf-8'))
            self.assertEqual(p['cursor_groups'], 2)
            build_features(root / 'resumed', identity, block, block.keys, model.encoder, c, PROTOCOL, 'validation')
            build_features(root / 'full', identity, block, block.keys, model.encoder, c, PROTOCOL, 'validation')
            a, b = FeatureCache(root / 'resumed'), FeatureCache(root / 'full')
            try:
                for name in a.arrays:
                    np.testing.assert_array_equal(a.arrays[name], b.arrays[name])
                target, _ = trainer.parent_train.prepare_training_targets(block.keys, PROTOCOL)
                np.testing.assert_array_equal(a.arrays['targets'][:, 0], target)
                self.assertGreater(np.nanstd(a.arrays['targets'][:, 0]), .9)
            finally:
                a.close(); b.close()
            with (root / 'resumed/features.npy').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                FeatureCache(root / 'resumed')

    def test_resume_inside_clock_and_between_clocks_is_exact(self):
        c = small_config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, validation = ArrayCache(root / 'train', 'train', 16), ArrayCache(root / 'validation', 'validation', 16)
            paths = [root / name for name in ('full', 'mid', 'boundary')]
            for p in paths:
                p.mkdir()
            full, _ = model_from_parent(c)
            fit.fit_readouts(full, train, validation, c, PROTOCOL, paths[0], torch.device('cpu'))
            step = fit.clock_training_step
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise KeyboardInterrupt
                return step(*args, **kwargs)
            mid, _ = model_from_parent(c)
            with patch.object(fit, 'clock_training_step', interrupt), self.assertRaises(KeyboardInterrupt):
                fit.fit_readouts(mid, train, validation, c, PROTOCOL, paths[1], torch.device('cpu'))
            mid, _ = model_from_parent(c)
            fit.fit_readouts(mid, train, validation, c, PROTOCOL, paths[1], torch.device('cpu'), resume=True)
            def boundary(event, **fields):
                if event == 'clock_completed' and fields['clock_id'] == 0:
                    raise KeyboardInterrupt
            between, _ = model_from_parent(c)
            with self.assertRaises(KeyboardInterrupt):
                fit.fit_readouts(between, train, validation, c, PROTOCOL, paths[2], torch.device('cpu'), report=boundary)
            between, _ = model_from_parent(c)
            fit.fit_readouts(between, train, validation, c, PROTOCOL, paths[2], torch.device('cpu'), resume=True)
            for other in (mid, between):
                for k, value in full.state_dict().items():
                    torch.testing.assert_close(value, other.state_dict()[k], rtol=0, atol=0)
            a = pd.read_csv(paths[0] / 'history.csv').drop(columns=['seconds', 'validation_seconds'])
            for path in paths[1:]:
                pd.testing.assert_frame_equal(a, pd.read_csv(path / 'history.csv').drop(columns=['seconds', 'validation_seconds']), check_exact=True)
                last_a, last_b = [torch.load(p / 'clocks/7/last.pt', weights_only=True) for p in (paths[0], path)]
                torch.testing.assert_close(last_a['rng']['torch'], last_b['rng']['torch'], rtol=0, atol=0)
                for k, v in last_a['ema_state'].items():
                    torch.testing.assert_close(v, last_b['ema_state'][k], rtol=0, atol=0)
                for parameter, values in last_a['optimizer']['state'].items():
                    for k, v in values.items():
                        torch.testing.assert_close(v, last_b['optimizer']['state'][parameter][k], rtol=0, atol=0)
            s = json.loads((paths[0] / 'selection.json').read_text(encoding='utf-8'))
            self.assertIsNone(s['best_epoch'])
            self.assertEqual(len(s['clock_selections']), 8)
            self.assertTrue(all(x['best_validation_ic'] >= x['parent_validation_ic'] for x in s['clock_selections']))

    def test_no_improvement_keeps_parent_readouts_and_stops_at_four(self):
        c = small_config(); c['training']['epochs'] = 10
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = root / 'run'; run.mkdir()
            train, val = ArrayCache(root / 'train', 'train', 16), ArrayCache(root / 'val', 'validation', 16)
            model, _ = model_from_parent(c)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            def no_update(*args, **kwargs):
                return {'loss': 1., 'main_loss': 1., 'rank_loss': 1., 'expert_loss': 1., 'router_loss': 1.}
            with patch.object(fit, 'clock_training_step', no_update):
                result = fit.fit_readouts(model, train, val, c, PROTOCOL, run, torch.device('cpu'))
            self.assertTrue(all(x['kept_parent'] and x['epochs_completed'] == 4 for x in result['clock_selections']))
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_interrupt_while_committing_epoch_selection_recovers_exactly(self):
        c = small_config(); c['training']['epochs'] = 1
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, val = ArrayCache(root / 'train', 'train', 16), ArrayCache(root / 'val', 'validation', 16)
            full, resumed = root / 'full', root / 'resumed'; full.mkdir(); resumed.mkdir()
            first, _ = model_from_parent(c)
            fit.fit_readouts(first, train, val, c, PROTOCOL, full, torch.device('cpu'))
            save = fit.atomic_save
            interrupted = False
            def crash(path, payload):
                nonlocal interrupted
                state = payload.get('progress', {})
                active = state.get('active')
                if Path(path) == resumed / 'last.pt' and active and active['epoch'] == 2 and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt
                return save(path, payload)
            second, _ = model_from_parent(c)
            with patch.object(fit, 'atomic_save', crash), self.assertRaises(KeyboardInterrupt):
                fit.fit_readouts(second, train, val, c, PROTOCOL, resumed, torch.device('cpu'))
            second, _ = model_from_parent(c)
            fit.fit_readouts(second, train, val, c, PROTOCOL, resumed, torch.device('cpu'), resume=True)
            for k, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
