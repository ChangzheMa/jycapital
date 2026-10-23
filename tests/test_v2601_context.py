"""完整上下文、日期内目标批次与日期权重的检查。"""
from types import SimpleNamespace
import unittest

import torch
import numpy as np

from test_v2601 import (small_config, Block, Store, MultiFrequencyGRU, trainer,
                         training_view, prepare_rank_training_targets, activate_market)
from v2601_context import MarketDayPlan, collect_context, native_batches, market_training_batches, index_view
from v2601_temporal import attach_temporal_weights, training_calendar_normalizer, normalize_day_weights


class MarketContextTests(unittest.TestCase):
    def fixture(self):
        config=small_config()
        block=Block('2020-01',1)
        block.keys.loc[block.keys.instrument=='S5','vwap_return']=np.nan
        primary,_=trainer.prepare_training_targets(block.keys,Store.protocol)
        ranks,_=prepare_rank_training_targets(block.keys,Store.protocol)
        view=training_view(block,primary,ranks)
        view.market_plan=MarketDayPlan(view)
        model=MultiFrequencyGRU(config)
        return config,block,view,model

    def test_full_context_retains_unlabelled_stocks_and_no_label_reads(self):
        config,block,view,model=self.fixture()
        self.assertEqual(view.eligible.sum(),10)
        record=view.market_plan.record(2)
        self.assertEqual(record['context_stock_days'],12)
        self.assertEqual(record['maximum_context_stocks'],6)
        self.assertEqual(record['optimizer_steps_per_full_pass'],6)
        groups=view.market_plan.groups[0]
        class ForbiddenTargets:
            def __getitem__(self,indices):raise AssertionError('context read labels')
        saved=view.targets;view.targets=ForbiddenTargets()
        try:
            state=torch.get_rng_state().clone()
            snapshots,members=collect_context(model,view,groups,native_batches,2,torch.device('cpu'))
            torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
        finally:view.targets=saved
        self.assertEqual(snapshots.shape,(6,8,1350))
        self.assertTrue(members.all())
        self.assertFalse(snapshots.is_inference())
        self.assertFalse(snapshots.requires_grad)
        # 即使改掉全部标签，行情与当前成员构成的上下文也不改变。
        block.keys['vwap_return']=np.nan
        other=index_view(block)
        second,mask=collect_context(model,other,groups,native_batches,3,torch.device('cpu'))
        torch.testing.assert_close(snapshots,second,rtol=0,atol=0)
        torch.testing.assert_close(members,mask,rtol=0,atol=0)

    def test_multiple_batches_per_date_and_resume_boundaries(self):
        config,block,view,model=self.fixture()
        order=view.market_plan.training_order(42,1,'2020-01')
        before=torch.get_rng_state().clone()
        complete=list(market_training_batches(model,view,order,view.targets,0,2,torch.device('cpu'),config,native_batches))
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
        self.assertEqual([len(item[1]) for item in complete],[2,2,1,2,2,1])
        self.assertEqual([item[0] for item in complete],[0,2,4,5,7,9])
        for offset,selected,windows,clock,target in complete:
            self.assertEqual(len(np.unique(view.index.dates[selected])),1)
            self.assertEqual(windows['market_snapshots'].shape[0],6)
            self.assertTrue(view.eligible[selected].all())
        resumed=list(market_training_batches(model,view,order,view.targets,5,2,torch.device('cpu'),config,native_batches))
        self.assertEqual([item[0] for item in resumed],[5,7,9])
        for left,right in zip(complete[3:],resumed):
            np.testing.assert_array_equal(left[1],right[1])
            for key in left[2]:torch.testing.assert_close(left[2][key],right[2][key],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'游标'):
            list(market_training_batches(model,view,order,view.targets,1,2,torch.device('cpu'),config,native_batches))
        mixed=order.copy();mixed[[1,5]]=mixed[[5,1]]
        with self.assertRaisesRegex(ValueError,'连续'):
            list(view.market_plan.segments(mixed,0,2))

    def test_global_calendar_normalization_preserves_relative_date_weights(self):
        config,block,view,model=self.fixture()
        source=SimpleNamespace(days=np.unique(block.keys.date.to_numpy(dtype='datetime64[ns]').astype('datetime64[D]')))
        record=training_calendar_normalizer(source,config,Store.protocol)
        attach_temporal_weights(view,config,Store.protocol)
        raw=view.sample_weights.copy()
        normalize_day_weights(view,record)
        np.testing.assert_allclose(view.sample_weights/raw,np.full(len(raw),1/record['normalizer']),rtol=0,atol=1e-15)
        self.assertNotEqual(view.sample_weights[0],view.sample_weights[-1])
        self.assertAlmostEqual(view.sample_weights.mean(),1.)
        with self.assertRaisesRegex(ValueError,'一次'):normalize_day_weights(view,record)

    def test_date_multiplier_scales_training_objective_not_cancelled(self):
        from test_v2601 import CONFIG
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__();self.scores=torch.nn.Parameter(torch.zeros(1,8,2))
            def forward_training_group(self,windows,clock):
                detail={'membership':torch.ones(1,8,dtype=torch.bool),'correction':torch.zeros_like(self.scores),
                        'modulation_scale_rms':torch.tensor(0.),'modulation_shift_rms':torch.tensor(0.)}
                return self.scores,self.scores[:,:,None,:].expand(-1,-1,4,-1),torch.zeros(1,8,4),detail
        model=Constant();target=torch.full((1,8,2),float('nan'),dtype=torch.float64);target[0,1]=torch.tensor([3.,1.])
        optimizer=torch.optim.SGD(model.parameters(),lr=0)
        first=trainer.training_step(model,optimizer,{},torch.zeros(1,8),target,CONFIG,False,torch.ones(1))
        second=trainer.training_step(model,optimizer,{},torch.zeros(1,8),target,CONFIG,False,torch.full((1,),2.))
        self.assertAlmostEqual(second['loss'],first['loss']*2)
        self.assertEqual(second['raw_main_loss'],first['raw_main_loss'])
        self.assertEqual(second['date_multiplier'],2.)


if __name__=='__main__':
    unittest.main()
