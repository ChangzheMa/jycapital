"""数量单位反变换的可信边界、缺失、极值与父缓存复用。"""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

from test_v2901 import small_config
from test_v2401_cache import fixture
from price_geometry import FIELDS
from price_memory_inputs import prepare_price_memory_windows
import price_memory_resident as parent
from quantity_units_inputs import (profiles_for_stocks,prepare_quantity_windows,QUANTITY_INDICES,FREQUENCIES)
from quantity_units_resident import QuantityBlock,ResidentBatches
from v2901_data import WindowBlock


def statistics():
    values={'count':[40]*36,'mean':[2.]*36,'m2':[160.]*36}
    second=copy.deepcopy(values)
    second['count'][QUANTITY_INDICES[1]]=0
    second['m2'][QUANTITY_INDICES[1]]=0.
    return {'fields':FIELDS,'stocks':{'S0':values,'S1':second}}


POLICY={'min_observations':30,'scale_epsilon':1e-8}


class QuantityUnitsTests(unittest.TestCase):
    def test_fixed_profile_trust_unknown_and_constant_fields(self):
        data=statistics()
        data['stocks']['S0']['m2'][QUANTITY_INDICES[0]]=0.
        profiles=profiles_for_stocks(data,POLICY,['S0','S1','NEW'])
        np.testing.assert_array_equal(profiles[2],np.zeros((22,2),np.float32))
        np.testing.assert_array_equal(profiles[0,0],[0,0])
        np.testing.assert_array_equal(profiles[1,1],[0,0])
        np.testing.assert_array_equal(profiles[0,1],[2,2])

    def test_inverse_zero_partial_missing_overflow_and_future_keep_parent(self):
        config=small_config(resident=True)
        raw,repairs={},{}
        for f,length in config['data']['window_bars'].items():
            codes=torch.zeros(1,length,72,dtype=torch.float16)
            codes[0,0,4]=-1.
            codes[0,2,4]=1.
            codes[0,3,16]=float('nan')
            codes[0,0,6]=float('nan')  # 价格缺失不清除同Bar有效数量。
            codes[0,4,4]=float('inf')
            profile=torch.from_numpy(profiles_for_stocks(statistics(),POLICY,['S0']))
            profile[0,1]=0  # 金额统计不可逆，原z仍保留。
            codes[0,1,5]=5.
            raw[f]=(codes,torch.full((1,length),2.),profile)
            repairs[f]=(torch.tensor([4*72+4]),torch.tensor([100000.]))
        clock=torch.eye(8)[[0]]
        result=prepare_quantity_windows(raw,repairs,clock)
        original=prepare_price_memory_windows({f:p[:2] for f,p in raw.items()},repairs,clock)
        for f,x in result.items():
            torch.testing.assert_close(x[:,:77],original[f],rtol=0,atol=0)
            self.assertEqual(x[0,77,0],0.)
            self.assertEqual(x[0,99,0],1.)
            self.assertAlmostEqual(float(x[0,77,2]),.4,places=6)
            self.assertEqual(x[0,73,0],0.)
            self.assertEqual(x[0,101,0],1.)
            self.assertEqual(x[0,101,3],0.)
            self.assertEqual(x[0,5,1],5.)
            self.assertEqual(x[0,100,1],0.)
            torch.testing.assert_close(x[0,77,4],torch.tensor(20000.2),rtol=0,atol=0)
            bars=240//int(f[3:-1])
            self.assertEqual(x[:,77:,-bars:].count_nonzero().item(),0)

    def test_parent_cache_native_stream_resident_and_prefetch_are_equal(self):
        with tempfile.TemporaryDirectory() as temporary:
            store,directory,key,manifest,native,_=fixture(Path(temporary))
            cache,cache_key=parent.ensure_side_cache(store,directory,key,manifest,split='train')
            preserved={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
            all_stats={f:statistics() for f in FREQUENCIES}
            profiles={f:profiles_for_stocks(all_stats[f],POLICY,['S0','S1']) for f in FREQUENCIES}
            reference=WindowBlock(native.keys,native.price_grids,pd.Index(['S0','S1']),small_config(resident=True),
                                  pd.to_datetime(['2020-01-02','2020-01-03']).to_numpy(),profiles)
            rows=np.arange(len(native.keys))
            expected,clock=reference.batch(rows)
            for resident in (True,False):
                block=QuantityBlock(parent.ResidentBlock(cache,cache_key,2.,resident=resident),all_stats,POLICY)
                try:
                    actual,actual_clock=block.batch(rows)
                    for f in FREQUENCIES:np.testing.assert_array_equal(actual[f],expected[f])
                    np.testing.assert_array_equal(actual_clock,clock)
                    order=rows[::-1].copy();rng=torch.get_rng_state().clone()
                    for start,chosen,windows,clocks,targets in ResidentBatches(block,order,block.targets,3,5,'cpu'):
                        values,known_clock=block.batch(order[start:start+5])
                        for f in FREQUENCIES:np.testing.assert_array_equal(windows[f].numpy(),values[f])
                        np.testing.assert_array_equal(clocks.numpy(),known_clock)
                        np.testing.assert_array_equal(targets.numpy(),block.targets[chosen])
                    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
                finally:block.close()
            self.assertEqual(preserved,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()})


if __name__=='__main__':unittest.main()
