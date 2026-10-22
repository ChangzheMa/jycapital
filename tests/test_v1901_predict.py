"""完整市场因果快路径：日内部分请求必须保留此前的股票池上下文。"""

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

from test_v1901 import ROOT,small_config,activate_prior
from test_v2_predict import FakeDai
from v1901_data import regular_times
from v1901_preprocessing import FIELDS,FREQUENCIES,pack_moments,moments_from_values
from v1901_model import MultiFrequencyGRU
from v1901_predict_fast import predict
from v1901_predict_reference import predict as predict_reference


class MarketDai(FakeDai):
    def query(self,sql,filters=None):
        if 'CAST(date AS DATE)' not in sql:
            return super().query(sql,filters)
        self.queries.append(sql)
        frame=self.tables['cpt_jyc_2026_instruments']
        stop=pd.Timestamp(re.search(r"date < '([^']+)'",sql).group(1))
        limit=int(re.search(r'LIMIT (\d+)',sql).group(1))
        days=sorted(frame.loc[frame.date<stop,'date'].dt.normalize().unique(),reverse=True)[:limit]
        frame=pd.DataFrame({'date':days})
        return type('Query',(),{'df':lambda _:frame})()


class MarketPredictTests(unittest.TestCase):
    def test_intraday_subrequest_keeps_earlier_exited_stock_context(self):
        config=small_config()
        preprocessing=json.loads((ROOT/'solutions/v1901/config.json').read_text(encoding='utf-8'))
        days=pd.bdate_range('2024-01-02',periods=4)
        minutes=(570,600,630,660,690,810,840,870)
        rows=[(day+pd.Timedelta(minutes=m),stock) for day in days for m in minutes
              for stock in (('A','B','C') if m<630 else ('A','C','NEW'))]
        tables={'cpt_jyc_2026_instruments':pd.DataFrame(rows,columns=['date','instrument'])}
        rng=np.random.default_rng(1501)
        stocks=('A','B','C','NEW')
        for frequency in FREQUENCIES:
            times=pd.to_datetime(regular_times(days,frequency))
            frame=pd.DataFrame([(t,stock) for t in times for stock in stocks],columns=['date','instrument'])
            for field in FIELDS:
                frame[field]=rng.uniform(1,20,len(frame))
            tables['fixture_'+frequency]=frame
        fixed=pack_moments(moments_from_values(np.stack([np.zeros(36),np.ones(36)]*40)))
        statistics={'fields':FIELDS,'fit_interval':{'start':'2020-01-01','stop':'2024-01-01'},
                    'stocks':{stock:fixed for stock in stocks if stock!='NEW'},'global':fixed}
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            (directory/'normalization').mkdir()
            for frequency in FREQUENCIES:
                (directory/'normalization'/f'{frequency}.json').write_text(json.dumps(statistics),encoding='utf-8')
            (directory/'preprocessing-config.json').write_text(json.dumps(preprocessing),encoding='utf-8')
            torch.manual_seed(42)
            model=MultiFrequencyGRU(config)
            activate_prior(model)
            torch.save({'config':config,'state_dict':model.state_dict()},directory/'best.pt')
            dai=MarketDai(tables)
            sources={frequency:'fixture_'+frequency for frequency in FREQUENCIES}
            whole=predict(sources,'2024-01-05','2024-01-05',directory,dai,device='cpu')
            part=predict(sources,'2024-01-05 10:30','2024-01-05 11:30',directory,dai,device='cpu')
            reference=predict_reference(sources,'2024-01-05','2024-01-05',directory,MarketDai(tables),device='cpu')
            np.testing.assert_allclose(whole.factor,reference.factor,rtol=1e-5,atol=2e-6)
        expected=whole.loc[whole.date.between('2024-01-05 10:30','2024-01-05 11:30')].reset_index(drop=True)
        pd.testing.assert_frame_equal(part[['date','instrument']],expected[['date','instrument']])
        np.testing.assert_allclose(part.factor,expected.factor,rtol=1e-5,atol=2e-6)
        self.assertEqual(len(whole),24)
        self.assertEqual(len(part),9)
        self.assertNotIn('B',part.instrument.unique())
        self.assertIn('B',whole.instrument.unique())
        self.assertEqual(part.factor.dtype,np.float64)
        self.assertTrue(np.isfinite(part.factor).all())
        self.assertFalse(any('vwap' in query or 'exposure' in query for query in dai.queries))


if __name__=='__main__':
    unittest.main()
