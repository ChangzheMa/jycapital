"""长期历史边界、初始退回父模型、非零历史读出及缓存复用。"""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

from test_v1701 import ROOT, CONFIG, PROTOCOL, small_config, inputs
sys.path.insert(0, str(ROOT / "solutions/v1102"))
from v1102_model import MultiFrequencyGRU as ParentModel
from v1701_model import MultiFrequencyGRU
from v1701_data import BaseProcessedStore, regular_times
from long_history import (LongHistory, LONG_KEY, VALUE_COLUMNS, history_positions, prepare_values,
                          from_grid, input_identity, append_long_batches)
from artifacts import write_json, sha256
from v1701_platform_history import DailyLongHistory


class LongContextTests(unittest.TestCase):
    def test_platform_daily_buffer_evicts_only_unneeded_dates(self):
        config = small_config()
        days = pd.bdate_range('2024-01-02',periods=12)
        times = regular_times(days,'bar30m')
        raw = np.arange(len(times)*2*72,dtype=np.float32).reshape(len(times),2,72)
        source = DailyLongHistory(['A','B'],days,config)
        for i,day in enumerate(days):
            source.append(day,raw[i*8:(i+1)*8])
            self.assertLessEqual(len(source.frames),9)
        keys = pd.DataFrame({'date':[days[-1]+pd.Timedelta(hours=10)]*2,'instrument':['A','B']})
        np.testing.assert_array_equal(source.batch(keys),from_grid((times,raw),['A','B'],days,keys,config))
        with self.assertRaises(ValueError):
            source.append(days[-1],raw[-8:])

    def test_completed_day_window_ignores_current_and_future_bars(self):
        config = small_config()
        days = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        times = regular_times(days, "bar30m")
        values = np.arange(len(times) * 2 * 72, dtype=np.float32).reshape(len(times), 2, 72)
        keys = pd.DataFrame({"date": pd.to_datetime(["2020-01-06 09:30", "2020-01-06 14:30"]), "instrument": ["A", "A"]})
        actual = from_grid((times, values), ["A", "B"], days, keys, config)
        np.testing.assert_array_equal(actual[0], actual[1])
        self.assertEqual(np.count_nonzero(actual[:, :, :6 * 8]), 0)
        np.testing.assert_array_equal(actual[0, :72, -16:], values[:16, 0].T)
        changed = values.copy()
        changed[16:] = 1e8
        np.testing.assert_array_equal(actual, from_grid((times, changed), ["A", "B"], days, keys, config))
        np.testing.assert_array_equal(history_positions(days, [days[0]], 3), [[-3, -2, -1]])

    def test_precision_overflow_and_missing_flags(self):
        raw = np.ones((1, 2, 8, 72), np.float32)
        raw[:, 0] = np.nan
        raw[0, 1, 0, 0] = 70000
        raw[0, 1, 0, 1] = 1.0001
        result = prepare_values(raw, "fp16_with_fp32_overflow")
        self.assertEqual(result.shape, (1, 75, 16))
        self.assertEqual(np.count_nonzero(result[:, :, :8]), 0)
        self.assertEqual(result[0, 0, 8], 70000)
        self.assertEqual(result[0, 1, 8], np.float16(1.0001))
        self.assertTrue((result[0, 72:, 8:] == 1).all())

    def test_initial_outputs_equal_parent_and_empty_history_stays_zero(self):
        config = small_config()
        parent_config = copy.deepcopy(config)
        parent_config['solution_id'] = 'v1102'
        torch.manual_seed(42)
        parent = ParentModel(parent_config).eval()
        torch.manual_seed(42)
        model = MultiFrequencyGRU(config).eval()
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        windows, clock = inputs(config)
        short = {k: v for k, v in windows.items() if k != LONG_KEY}
        with torch.inference_mode():
            expected = parent.forward_group(short, clock)
            torch.testing.assert_close(model.forward_group(windows, clock), expected, rtol=0, atol=0)
            model.long_history.output.weight.normal_(std=.02)
            model.long_history.output.bias.fill_(.1)
            windows[LONG_KEY].zero_()
            windows[LONG_KEY][:, :72] = 999
            torch.testing.assert_close(model.forward_group(windows, clock), expected, rtol=0, atol=0)

    def test_nonzero_history_is_learnable_and_shared_all_clock_readout_matches(self):
        config = small_config()
        model = MultiFrequencyGRU(config).eval()
        torch.nn.init.normal_(model.long_history.output.weight, std=.02)
        windows, clock = inputs(config)
        windows[LONG_KEY].requires_grad_(True)
        shared = model.forward_group(windows, clock)
        for current in range(8):
            selected = torch.nn.functional.one_hot(torch.full((len(clock),), current), 8).float()
            reference = model.forward_heads(windows, selected)
            torch.testing.assert_close(shared[:, current], reference, rtol=1e-5, atol=2e-7)
        gradient = torch.autograd.grad(shared.sum(), windows[LONG_KEY])[0]
        self.assertGreater(gradient[:, :72].abs().sum().item(), 0)

    def test_parent_evaluation_cache_identity_is_reused(self):
        reference = importlib.import_module('v1102_evaluation_cache')
        current = importlib.import_module('v1701_evaluation_cache')
        for name in ('v1102_resident.py', 'v1102_evaluation_cache.py'):
            self.assertEqual((ROOT/'solutions/v1102'/name).read_bytes(), (ROOT/'solutions/v1701'/name).read_bytes())
        old_data = importlib.import_module('v1102_data')
        before = old_data.ProcessedStore.__new__(old_data.ProcessedStore)
        before.config = json.loads((ROOT/'solutions/v1102/model-config.json').read_text(encoding='utf-8'))
        before.protocol = PROTOCOL
        after = BaseProcessedStore.__new__(BaseProcessedStore)
        after.config, after.protocol = CONFIG, PROTOCOL
        for split in ('validation', 'test'):
            self.assertEqual(reference.cache_key(before, split, {'synthetic': True}),
                             current.cache_key(after, split, {'synthetic': True}))

    def test_prefetch_resume_preserves_order_and_does_not_consume_rng(self):
        order = np.array([4, 2, 1, 0, 3, 5])
        representatives = np.array([10, 20, 30, 40, 50, 60])
        class Block:
            def long_batch(self, rows):
                return np.broadcast_to(rows[:, None, None], (len(rows), 75, 8)).astype(np.float32).copy()
        def short_iterator():
            for offset in range(2, len(order), 2):
                selected = order[offset:offset + 2]
                yield offset, selected, {}, torch.zeros(len(selected), 8), torch.zeros(len(selected))
        before = torch.get_rng_state().clone()
        result = list(append_long_batches(short_iterator(), Block(), order, 2, 2, torch.device('cpu'), representatives))
        self.assertEqual([item[0] for item in result], [2, 4])
        for offset, selected, windows, clock, targets in result:
            np.testing.assert_array_equal(windows[LONG_KEY][:, 0, 0].numpy(), representatives[selected])
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)

    def test_cache_matches_native_grid_without_any_label_table(self):
        config = small_config()
        protocol = copy.deepcopy(PROTOCOL)
        protocol['splits'] = {'train': {'start':'2020-01-01','stop':'2020-02-01'},
                              'validation': {'start':'2020-02-01','stop':'2020-03-01'},
                              'test': {'start':'2020-03-01','stop':'2020-04-01'}}
        days = pd.to_datetime(['2020-01-02','2020-01-03','2020-02-03','2020-03-02'])
        stocks = pd.Index(['A','B'])
        universe = pd.DataFrame([(d+pd.Timedelta(minutes=m),s) for d in days for m in (570,600,630,660,690,810,840,870)
                                 for s in stocks], columns=['date','instrument'])
        times = regular_times(days, 'bar30m')
        raw = np.arange(len(times)*2*72,dtype=np.float32).reshape(len(times),2,72) / 10
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, training, holdout = (root/n for n in ('data','train','holdout'))
            for directory in (data, training, holdout): directory.mkdir()
            universe.to_parquet(data/(protocol['universe_table']+'.parquet'),index=False)
            for directory in (training,holdout): write_json(directory/'manifest.json',{'status':'complete','fixture':True})
            for split, interval in protocol['splits'].items():
                selected = (times >= pd.Timestamp(interval['start']).value) & (times < pd.Timestamp(interval['stop']).value)
                part_times = times[selected]
                frame = pd.DataFrame([(pd.Timestamp(t),s) for t in part_times for s in stocks],columns=['date','instrument'])
                for i,column in enumerate(VALUE_COLUMNS): frame[column] = raw[selected,:,i].reshape(-1)
                prepared = training if split=='train' else holdout
                relative = Path('bar30m') if split=='train' else Path(split)/'bar30m'
                path = prepared/'data'/relative/(str(pd.Timestamp(interval['start']).to_period('M'))+'.parquet')
                path.parent.mkdir(parents=True)
                frame.to_parquet(path,index=False)
                write_json(prepared/'records'/relative/(path.stem+'.json'),{'bytes':path.stat().st_size,'sha256':sha256(path)})
            store = BaseProcessedStore(training,holdout,data,config,protocol)
            for split, interval in protocol['splits'].items():
                reader = LongHistory(store,split)
                try:
                    keys = universe.loc[(universe.date>=interval['start']) & (universe.date<interval['stop'])]
                    actual = reader.batch(keys.date.to_numpy(),keys.instrument.to_numpy())
                    expected = from_grid((times,raw),stocks,days,keys,config)
                    np.testing.assert_array_equal(actual,expected)
                    self.assertFalse((data/(protocol['label_table']+'.parquet')).exists())
                    key, _ = input_identity(store,split)
                    alternate = copy.copy(store)
                    alternate.config = copy.deepcopy(config)
                    alternate.config['model']['hidden_dim'] *= 2
                    self.assertEqual(input_identity(alternate,split)[0],key)
                    alternate.config['long_context']['previous_days'] *= 2
                    self.assertNotEqual(input_identity(alternate,split)[0],key)
                    source_record = training/'records/bar30m/2020-01.json'
                    saved_record = json.loads(source_record.read_text(encoding='utf-8'))
                    write_json(source_record,dict(saved_record,sha256='different-source'))
                    self.assertNotEqual(input_identity(store,split)[0],key)
                    write_json(source_record,saved_record)
                    directory = reader.directory
                finally:
                    reader.close()
            values_path = directory/'values.npy'
            with values_path.open('r+b') as stream:
                stream.seek(-1,2)
                value=stream.read(1)
                stream.seek(-1,2)
                stream.write(bytes([value[0]^1]))
            with self.assertRaises(ValueError): LongHistory(store,'test')


if __name__ == '__main__':
    unittest.main()
