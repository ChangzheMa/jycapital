"""长期平台输入：超过短期窗口的历史、fast参考和请求拆分一致。"""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

from test_v1701 import ROOT, small_config
from test_v2_predict import FakeDai
from v1701_data import regular_times
from v1701_preprocessing import FIELDS, FREQUENCIES, pack_moments, moments_from_values
from v1701_model import MultiFrequencyGRU
from v1701_predict import predict
from v1701_predict_reference import predict as reference_predict


class CalendarDai(FakeDai):
    def query(self, sql, filters=None):
        if 'CAST(date AS DATE)' not in sql:
            return super().query(sql, filters)
        self.queries.append(sql)
        frame = self.tables['cpt_jyc_2026_instruments']
        stop = pd.Timestamp(re.search(r"date < '([^']+)'", sql).group(1))
        limit = int(re.search(r'LIMIT (\d+)', sql).group(1))
        days = sorted(frame.loc[frame.date < stop, 'date'].dt.normalize().unique(), reverse=True)[:limit]
        return type('Query', (), {'df': lambda _: pd.DataFrame({'date': days})})()


class LongPredictTests(unittest.TestCase):
    def test_nonzero_long_history_fast_reference_split_and_old_history_effect(self):
        config = small_config()
        config['long_context']['previous_days'] = 24
        preprocessing = json.loads((ROOT/'solutions/v1701/config.json').read_text(encoding='utf-8'))
        days = pd.bdate_range('2024-01-02', periods=26)
        minutes = (570,600,630,660,690,810,840,870)
        stocks = ('A','B','C','NEW')
        universe = pd.DataFrame([(d+pd.Timedelta(minutes=m),s) for d in days for m in minutes
                                 for s in (('A','B','C') if m<630 else ('A','C','NEW'))], columns=['date','instrument'])
        tables = {'cpt_jyc_2026_instruments': universe}
        rng = np.random.default_rng(1701)
        for frequency in FREQUENCIES:
            times = pd.to_datetime(regular_times(days, frequency))
            frame = pd.DataFrame([(t,s) for t in times for s in stocks], columns=['date','instrument'])
            for field in FIELDS:
                frame[field] = rng.uniform(1,20,len(frame))
            tables['fixture_'+frequency] = frame
        fixed = pack_moments(moments_from_values(np.stack([np.zeros(36),np.ones(36)]*40)))
        statistics = {'fields': FIELDS, 'fit_interval': {'start':'2020-01-01','stop':'2024-01-01'},
                      'stocks': {stock:fixed for stock in stocks}, 'global':fixed}
        sources = {frequency:'fixture_'+frequency for frequency in FREQUENCIES}
        first, last = (str(day.date()) for day in days[-2:])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'normalization').mkdir()
            for frequency in FREQUENCIES:
                (directory/'normalization'/f'{frequency}.json').write_text(json.dumps(statistics),encoding='utf-8')
            (directory/'preprocessing-config.json').write_text(json.dumps(preprocessing),encoding='utf-8')
            torch.manual_seed(42)
            model = MultiFrequencyGRU(config)
            torch.nn.init.normal_(model.long_history.output.weight,std=.1)
            torch.nn.init.normal_(model.long_history.output.bias,std=.01)
            torch.save({'config':config,'state_dict':model.state_dict()},directory/'best.pt')
            dai = CalendarDai(tables)
            whole = predict(sources,first,last,directory,dai,device='cpu')
            whole_queries = list(dai.queries)
            reference = reference_predict(sources,first,last,directory,dai,device='cpu')
            split = pd.concat([predict(sources,day,day,directory,dai,device='cpu') for day in (first,last)],ignore_index=True)
            partial = predict(sources,last+' 10:30',last+' 11:30',directory,dai,device='cpu')
            for expected in (reference,split):
                pd.testing.assert_frame_equal(whole[['date','instrument']],expected[['date','instrument']])
                np.testing.assert_allclose(whole.factor,expected.factor,rtol=1e-4,atol=2e-7)
            expected = whole.loc[whole.date.between(last+' 10:30',last+' 11:30')].reset_index(drop=True)
            pd.testing.assert_frame_equal(partial[['date','instrument']],expected[['date','instrument']])
            np.testing.assert_allclose(partial.factor,expected.factor,rtol=1e-4,atol=2e-7)
            changed = {k:v.copy() for k,v in tables.items()}
            frame = changed['fixture_bar30m']
            # 此日处于24日历史内、20日短期窗口以外，且统计量均可信固定。
            selected = frame.date.dt.normalize().eq(days[2]) & frame.instrument.eq('A')
            frame.loc[selected,FIELDS] = 1000
            altered = predict(sources,last,last,directory,CalendarDai(changed),device='cpu')
            expected = whole.loc[whole.date.dt.normalize().eq(days[-1])].reset_index(drop=True)
            self.assertGreater(np.max(np.abs(altered.factor-expected.factor)),1e-9)
        self.assertEqual(len(whole),48)
        self.assertEqual(len(partial),9)
        self.assertEqual(whole.factor.dtype,np.float64)
        self.assertTrue(np.isfinite(whole.factor).all())
        self.assertFalse(whole.duplicated(['date','instrument']).any())
        self.assertFalse(any('vwap' in query or 'exposure' in query for query in dai.queries))
        raw_queries = [query for query in whole_queries if query.startswith('SELECT date, instrument, open') and
                       'FROM fixture_bar30m ' in query]
        self.assertLessEqual(len(raw_queries), 2)


if __name__ == '__main__':
    unittest.main()
