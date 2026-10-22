"""完整市场时空模型的因果性、成员语义、分块和恢复检查。"""

import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'solutions/v1501'))
from v1501_model import MultiFrequencyPatchTransformer
import v1501_train as trainer
from v1501_grouped import training_view,StockDayIndex
from v1501_temporal import attach_temporal_weights
from v1501_calendar import CalendarDayPlan
from v1501_objective import full_day_objective
from v1501_market import predict_complete_days
from test_v1301 import Block,Store

CONFIG=json.loads((ROOT/'solutions/v1501/model-config.json').read_text(encoding='utf-8'))
PROTOCOL=json.loads((ROOT/'evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))
CONFIGS={'v1501':CONFIG}
TEST_VERSIONS=('v1501',)
PARENTS={'v1501':'v1301'}
torch.set_num_threads(1)


def small_config(version='v1501',resident=True):
    config=copy.deepcopy(CONFIG)
    config['model'].update(hidden_dim=8,fusion_hidden_dim=16,encoding_stock_batch_size=2)
    config['training'].update(epochs=2,cpu_threads=1,stock_day_batch_size=8,batch_size=64,
                              microbatch_stock_days=2,checkpoint_batches=1)
    config['inference'].update(stock_day_batch_size=8,batch_size=64)
    config['training']['resident_cache'].update(memory_reserve_gib=0,build_batch_rows=256,evaluation_days=1)
    return config


def inputs(config,count=4):
    generator=torch.Generator().manual_seed(1501)
    windows={frequency:torch.randn(count,75,length,generator=generator) for frequency,length in config['data']['window_bars'].items()}
    for value in windows.values():
        value[:,72:]=1
    return windows,F.one_hot(torch.full((count,),7),8).float()


class MarketTransformerTests(unittest.TestCase):
    def test_default_architecture_and_cache_target_contract(self):
        trainer.validate_training(CONFIG,PROTOCOL)
        model=MultiFrequencyPatchTransformer(CONFIG)
        self.assertEqual(sum(p.numel() for p in model.parameters()),3141398)
        self.assertFalse(any(isinstance(module,torch.nn.GRU) for module in model.modules()))
        sys.path.insert(0,str(ROOT/'solutions/v1002'))
        import v1002_train
        previous=json.loads((ROOT/'solutions/v1002/model-config.json').read_text(encoding='utf-8'))
        self.assertEqual(CONFIG['data'],previous['data'])
        self.assertEqual(inspect.getsource(trainer.prepare_training_targets),inspect.getsource(v1002_train.prepare_training_targets))
        self.assertEqual((ROOT/'solutions/v1501/v1501_resident.py').read_bytes(),(ROOT/'solutions/v1002/v1002_resident.py').read_bytes())

    def test_all_stock_outputs_are_causal_at_every_prediction_clock(self):
        config=small_config()
        model=MultiFrequencyPatchTransformer(config).eval()
        windows,clock=inputs(config)
        windows={frequency:value.requires_grad_(True) for frequency,value in windows.items()}
        output=model.forward_group(windows,clock)
        for moment in range(8):
            gradients=torch.autograd.grad(output[:,moment].sum(),tuple(windows.values()),retain_graph=True)
            for frequency,gradient in zip(windows,gradients):
                bars=240//int(frequency[3:-1])
                cutoff=CONFIG['data']['trading_day_windows'][frequency]['previous_days']*bars+moment*(bars//8)
                self.assertEqual(gradient[:,:,cutoff:].count_nonzero().item(),0)
                self.assertGreater(gradient[:,:,:cutoff].abs().sum().item(),0)

    def test_future_members_do_not_affect_prior_market_outputs(self):
        config=small_config()
        model=MultiFrequencyPatchTransformer(config).eval()
        windows,clock=inputs(config)
        membership=torch.ones(4,8,dtype=torch.bool)
        membership[3,:4]=False
        with torch.no_grad():
            before=model.forward_group(windows,clock,membership)
            changed={key:value.clone() for key,value in windows.items()}
            for value in changed.values():
                value[3,:72]+=20
            after=model.forward_group(changed,clock,membership)
        torch.testing.assert_close(before[:3,:4],after[:3,:4],rtol=0,atol=0)
        self.assertEqual(before[3,:4].count_nonzero().item(),0)
        self.assertGreater((before[:3,4:]-after[:3,4:]).abs().max().item(),1e-7)

    def test_stock_permutation_and_encoding_chunk_size_preserve_market_context(self):
        config=small_config()
        model=MultiFrequencyPatchTransformer(config).eval()
        windows,clock=inputs(config,count=5)
        order=torch.tensor([3,0,4,2,1])
        with torch.no_grad():
            expected=model.forward_group(windows,clock)
            changed=model.forward_group({key:value[order] for key,value in windows.items()},clock[order])
            torch.testing.assert_close(changed,expected[order],rtol=1e-5,atol=2e-6)
            for chunk in (1,3,64):
                model.encoding_stock_batch_size=chunk
                torch.testing.assert_close(model.forward_group(windows,clock),expected,rtol=1e-5,atol=2e-6)

    def test_unlabelled_stocks_remain_in_training_context(self):
        block=Block('2020-01',1)
        block.keys.loc[block.keys.instrument.eq('S5'),'vwap_return']=np.nan
        primary,_=trainer.prepare_training_targets(block.keys,Store.protocol)
        ranks,_=trainer.prepare_rank_training_targets(block.keys,Store.protocol)
        view=training_view(block,primary,ranks)
        attach_temporal_weights(view,CONFIG,Store.protocol)
        plan=CalendarDayPlan(view,CONFIG,Store.protocol)
        self.assertEqual([len(groups) for groups in plan.groups],[6,6,6])
        self.assertEqual(plan.record['context_stock_days'],18)
        self.assertEqual(plan.record['supervised_stock_days'],15)
        self.assertTrue((view.index.samples>=0).all())
        self.assertEqual(np.isfinite(view.targets.primary).sum(),105)

    def test_standalone_loss_masks_labels_without_removing_input_members(self):
        prediction=torch.randn(4,8,2,dtype=torch.float64,requires_grad=True)
        target=torch.randn(4,8,2,dtype=torch.float64)
        target[1,3]=float('nan')
        first,details=full_day_objective(prediction,target,torch.ones(4),CONFIG['training'])
        second,_=full_day_objective(prediction,target,torch.full((4,),2.),CONFIG['training'])
        torch.testing.assert_close(second,2*first,rtol=0,atol=0)
        first.backward()
        self.assertEqual(prediction.grad[1,3].count_nonzero().item(),0)
        self.assertEqual(details['rows'],31)

    def test_encoder_recomputation_matches_retained_graph_with_dropout(self):
        config=small_config()
        windows,clock=inputs(config,count=3)
        targets=torch.randn(3,8,2,dtype=torch.float64)
        for amp in (False,True):
            torch.manual_seed(7)
            models=[MultiFrequencyPatchTransformer(config).train()]
            models.append(copy.deepcopy(models[0]))
            models[0].activation_checkpoint=False
            initial=torch.get_rng_state().clone()
            records=[]
            for model in models:
                torch.set_rng_state(initial)
                with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
                    predictions=model.forward_group(windows,clock)
                loss,_=full_day_objective(predictions,targets,None,config['training'])
                loss.backward()
                records.append((predictions.detach(),torch.get_rng_state().clone()))
            torch.testing.assert_close(records[0][0],records[1][0],rtol=0,atol=0)
            torch.testing.assert_close(records[0][1],records[1][1],rtol=0,atol=0)
            for left,right in zip(models[0].parameters(),models[1].parameters()):
                torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)

    def test_interrupted_full_market_training_restores_exact_state(self):
        config,store=small_config(),Store()
        configure,step=trainer.configure,trainer.training_step
        with tempfile.TemporaryDirectory() as temporary:
            runs=[Path(temporary)/name for name in ('continuous','resumed')]
            manifests=[{'status':'running'},{'status':'running'}]
            for run in runs:
                run.mkdir()
                trainer.write_json(run/'data-manifest.json',{'synthetic':True})
                trainer.write_json(run/'baseline-preservation.json',{})
            def cpu(spec):
                configure(spec)
                return torch.device('cpu')
            with patch.object(trainer,'configure',cpu),patch.object(trainer,'environment',lambda:{'packages':{}}),\
                    patch.object(trainer,'draw_history'),patch.object(trainer,'log_event'):
                trainer.train(store,runs[0],config,manifests[0])
                calls=0
                def interrupt(*args,**kwargs):
                    nonlocal calls
                    calls+=1
                    if calls==3:
                        raise KeyboardInterrupt
                    return step(*args,**kwargs)
                with patch.object(trainer,'training_step',interrupt),self.assertRaises(KeyboardInterrupt):
                    trainer.train(store,runs[1],config,manifests[1])
                saved=torch.load(runs[1]/'last.pt',weights_only=True)
                self.assertEqual(saved['progress']['day_cursor'],2)
                self.assertEqual(saved['progress']['group_cursor'],12)
                trainer.train(store,runs[1],config,manifests[1],resume=True)
            for name in ('best.pt','last.pt'):
                left,right=[torch.load(run/name,weights_only=True) for run in runs]
                for key,value in left['state_dict'].items():
                    torch.testing.assert_close(value,right['state_dict'][key],rtol=0,atol=0)
            for key,value in left['ema_state_dict'].items():
                torch.testing.assert_close(value,right['ema_state_dict'][key],rtol=0,atol=0)
            for parameter,values in left['optimizer']['state'].items():
                for key,value in values.items():
                    torch.testing.assert_close(value,right['optimizer']['state'][parameter][key],rtol=0,atol=0)
            torch.testing.assert_close(left['torch_rng'],right['torch_rng'],rtol=0,atol=0)
            self.assertEqual(left['ema_stock_days'],36)
            self.assertEqual(left['ema_stock_days'],right['ema_stock_days'])
            frames=[pd.read_csv(run/'history.csv') for run in runs]
            columns=['train_loss','validation_loss','ic_mean','rank_ic_mean','ema_stock_days']+['train_'+key for key in trainer.CALENDAR_STATISTICS]
            pd.testing.assert_frame_equal(frames[0][columns],frames[1][columns],check_exact=True)


if __name__=='__main__':
    unittest.main()
