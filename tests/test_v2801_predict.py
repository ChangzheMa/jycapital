"""真实预测入口的合成行情、动态股票池及树补充；不访问平台。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import numpy as np
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
for folder in ('common','evaluation','solutions/v2801'):
    sys.path.insert(0,str(ROOT/folder))
from test_v2401_predict import MarketDai
from v2401_data import regular_times
from v2401_preprocessing import FIELDS,FREQUENCIES,pack_moments,moments_from_values
from v2801_model import MultiFrequencyNeuralTree
from v2801_predict import predict
from v2801_predict_reference import predict as reference
from v2801_text_artifact import FORMAT,encode_state,decode_state
import pandas as pd


class PredictTests(unittest.TestCase):
    def test_dynamic_pool_partial_request_and_text_tree_state(self):
        config=json.loads((ROOT/'solutions/v2801/model-config.json').read_text(encoding='utf-8'))
        parent=copy.deepcopy(config['backbone_config'])
        parent['model'].update(hidden_dim=8,fusion_hidden_dim=16,expert_hidden_dim=8,router_hidden_dim=8,
            book_level_dim=4,book_mixer_hidden_dim=4,non_book_dim=4)
        parent['model']['price_memory'].update(width=8,heads=2,distance_hidden_dim=5)
        config['backbone_config']=parent;config['model']['feature_dim']=256
        config['training'].update(cpu_threads=1,boosting_rounds=4)
        config['training']['booster']['max_depth']=2
        config['inference']['stock_day_batch_size']=2
        preprocessing=json.loads((ROOT/'solutions/v2801/config.json').read_text(encoding='utf-8'))
        days=pd.bdate_range('2024-01-02',periods=4);minutes=(570,600,630,660,690,810,840,870)
        rows=[(day+pd.Timedelta(minutes=m),stock) for day in days for m in minutes
              for stock in (('A','B','C') if m<630 else ('A','C','NEW'))]
        tables={'cpt_jyc_2026_instruments':pd.DataFrame(rows,columns=['date','instrument'])}
        rng=np.random.default_rng(2801);stocks=('A','B','C','NEW')
        for frequency in FREQUENCIES:
            grid=pd.to_datetime(regular_times(days,frequency))
            frame=pd.DataFrame([(t,s) for t in grid for s in stocks],columns=['date','instrument'])
            for field in FIELDS:frame[field]=rng.uniform(1,20,len(frame))
            tables['fixture_'+frequency]=frame
        fixed=pack_moments(moments_from_values(np.stack([np.zeros(36),np.ones(36)]*40)))
        statistics={'fields':FIELDS,'fit_interval':{'start':'2020-01-01','stop':'2024-01-01'},
            'stocks':{s:fixed for s in stocks if s!='NEW'},'global':fixed}
        model=MultiFrequencyNeuralTree(config).eval()
        x=rng.normal(size=(128,256)).astype(np.float32)
        y=(x[:,0]>0).astype(np.float32)+.2*x[:,1]
        booster=xgb.train({'objective':'reg:squarederror','base_score':0.,'max_depth':2,'nthread':1},
                          xgb.DMatrix(x,label=y,base_margin=x[:,-1],nthread=1),4)
        model.trees.import_booster(booster.save_raw(raw_format='json'))
        encoded={'format':FORMAT,'state_dict':encode_state(model.state_dict())}
        decoded=decode_state(json.loads(json.dumps(encoded)))
        for name,value in model.state_dict().items():torch.testing.assert_close(value,decoded[name],rtol=0,atol=0)
        bundle={'format':FORMAT,'state_dict':encoded['state_dict'],'config':config,'preprocessing':preprocessing,
                'normalization':{f:statistics for f in FREQUENCIES}}
        with tempfile.TemporaryDirectory() as temporary:
            sources={f:'fixture_'+f for f in FREQUENCIES};dai=MarketDai(tables)
            whole=predict(sources,'2024-01-05','2024-01-05',temporary,dai,device='cpu',bundle=bundle)
            part=predict(sources,'2024-01-05 10:30','2024-01-05 11:30',temporary,dai,device='cpu',bundle=bundle)
            direct=reference(sources,'2024-01-05','2024-01-05',temporary,MarketDai(tables),device='cpu',bundle=bundle)
        np.testing.assert_allclose(whole.factor,direct.factor,atol=2e-6,rtol=1e-5)
        expected=whole.loc[whole.date.between('2024-01-05 10:30','2024-01-05 11:30')].reset_index(drop=True)
        pd.testing.assert_frame_equal(part[['date','instrument']],expected[['date','instrument']])
        np.testing.assert_allclose(part.factor,expected.factor,atol=2e-6,rtol=1e-5)
        self.assertEqual(len(whole),24);self.assertEqual(len(part),9)
        self.assertTrue(np.isfinite(part.factor).all());self.assertEqual(part.factor.dtype,np.float64)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(part.date))
        self.assertTrue(part.instrument.map(lambda x:isinstance(x,str)).all())
        self.assertFalse(part.duplicated(['date','instrument']).any())
        self.assertFalse(any('vwap' in q or 'exposure' in q for q in dai.queries))


if __name__=='__main__':unittest.main()
