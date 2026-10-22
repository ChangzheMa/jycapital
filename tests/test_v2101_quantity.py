"""数量目标和可部署历史输入隔离；成员、训练边界及稀疏极值。"""
from types import SimpleNamespace
import unittest

import torch
import numpy as np
import pandas as pd

from test_v2101 import small_config, PROTOCOL
from v2101_data import WindowBlock, regular_times
from v2101_grouped import training_view
from v2101_quantity import attach_quantity_targets, _resident_quantity, INDICES


def fixture():
    config=small_config()
    days=pd.to_datetime(['2020-06-01','2020-06-02'])
    clocks=[570,600,630,660,690,810,840,870]
    keys=pd.DataFrame([(days[-1]+pd.Timedelta(minutes=c),stock) for c in clocks for stock in ('A','B')],columns=['date','instrument'])
    keys=keys.loc[~((keys.instrument=='B') & (keys.date.dt.hour==10))].reset_index(drop=True)
    grids={}
    for frequency in config['data']['window_bars']:
        times=regular_times(days,frequency)
        values=np.arange(len(times)*2*72,dtype=np.float32).reshape(len(times),2,72)/100
        grids[frequency]=(times,values)
    block=WindowBlock(keys,grids,pd.Index(['A','B']),config,days)
    targets=np.arange(len(keys),dtype=np.float64)
    return block,training_view(block,targets,targets.copy())


class QuantityTargetTests(unittest.TestCase):
    def test_future_target_does_not_enter_current_input(self):
        block,view=fixture();attach_quantity_targets(view,PROTOCOL)
        a_group=int(np.flatnonzero(block.stock_indices[view.index.representatives]==0)[0])
        row=int(np.flatnonzero((block.keys.instrument=='A') & (block.keys.date.dt.hour==11) & (block.keys.date.dt.minute==30))[0])
        before,clock=block.batch([row]);targets=view.targets[[a_group]].copy()
        raw=block.grids['bar30m'][1]
        raw[-4,0,INDICES]+=999  # 13:30 Bar，在11:30为辅助未来目标。
        after,_=block.batch([row]);changed=view.targets[[a_group]]
        for frequency in before: np.testing.assert_array_equal(before[frequency],after[frequency])
        np.testing.assert_array_equal(changed[:,:,:2],targets[:,:,:2])
        np.testing.assert_allclose(changed[:,4,2:24]-targets[:,4,2:24],999,rtol=0,atol=1e-4)
        np.testing.assert_array_equal(changed[:,4,24:46],targets[:,4,24:46])

    def test_membership_and_holdout_rejection(self):
        block,view=fixture();attach_quantity_targets(view,PROTOCOL)
        b_group=int(np.flatnonzero(block.stock_indices[view.index.representatives]==1)[0])
        targets=view.targets[[b_group]]
        self.assertTrue(np.isnan(targets[:,1:3]).all())
        self.assertTrue(np.isfinite(targets[:,0]).all())
        block.times=pd.Timestamp('2024-06-01').value + block.times % 86400000000000
        another=training_view(block,np.ones(len(block.keys)),np.ones(len(block.keys)))
        with self.assertRaises(ValueError): attach_quantity_targets(another,PROTOCOL)

    def test_sparse_exception_and_missing_previous_day(self):
        values=np.zeros((2,8,72),np.float16);values[0]=np.nan;values[1,0,4]=np.inf
        base=SimpleNamespace(arrays={'bar30m-map':np.array([[1]],np.int32),'bar30m-values':values,
              'bar30m-exception-pointers':np.array([0,0,1]),'bar30m-exception-offsets':np.array([4]),
              'bar30m-exception-values':np.array([100000.],np.float32)})
        actual=_resident_quantity(base,np.array([0,-1]),np.array([0,0]))
        self.assertEqual(actual[0,0,0],100000.)
        self.assertTrue(np.isnan(actual[1]).all())
        self.assertTrue(np.isinf(values[1,0,4]))


if __name__=='__main__': unittest.main()
