"""未来数量监督、主任务优先梯度及因果收益预测的 CPU 合成检查。"""
import copy
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solutions/v2101'))
import v2101_train as trainer
from v2101_model import MultiFrequencyGRU
from v2101_objective import task_objectives, combine_gradients
from v2101_ema import WeightEMA

CONFIG = json.loads((ROOT/'solutions/v2101/model-config.json').read_text(encoding='utf-8'))
CONFIGS = {'v2101': CONFIG}
TEST_VERSIONS = ('v2101',)
PARENTS = {'v2101': 'v1601'}
PROTOCOL = json.loads((ROOT/'evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))
torch.set_num_threads(1)


def small_config(version='v2101', resident=False):
    config = copy.deepcopy(CONFIG)
    config['model'].update(hidden_dim=8, fusion_hidden_dim=16, book_level_dim=4, book_mixer_hidden_dim=4,
                           non_book_dim=4, task_hidden_dim=12, task_feature_dim=8, task_router_hidden_dim=8,
                           quantity_head_hidden_dim=8, return_head_hidden_dim=12)
    config['training'].update(epochs=2, cpu_threads=1, batch_size=16, stock_day_batch_size=2, checkpoint_batches=1)
    config['inference'].update(batch_size=16, stock_day_batch_size=2)
    config['training']['resident_cache'].update(memory_reserve_gib=0, build_batch_rows=256, evaluation_days=1)
    if not resident:
        config['training']['batching']='monthly_shuffled_stock_days'
        config['data']['input_precision']='float32'
    return config


def inputs(config, count=3):
    generator=torch.Generator().manual_seed(55)
    windows={f:torch.randn(count,75,n,generator=generator) for f,n in config['data']['window_bars'].items()}
    for value in windows.values(): value[:,72:]=1
    return windows,F.one_hot(torch.full((count,),7),8).float()


class QuantityModelTests(unittest.TestCase):
    def test_default_contract_and_parameter_budget(self):
        trainer.validate_training(CONFIG, PROTOCOL)
        model=MultiFrequencyGRU(CONFIG)
        self.assertEqual(sum(p.numel() for p in model.parameters()),3365542)
        self.assertEqual(CONFIG['training']['resident_cache']['memory_reserve_gib'],2)

    def test_all_clocks_future_gradient_and_reference(self):
        config=small_config();torch.manual_seed(13)
        model=MultiFrequencyGRU(config).eval()
        windows,latest=inputs(config,count=1)
        windows={f:x.requires_grad_() for f,x in windows.items()}
        grouped=model.forward_group_tasks(windows,latest)
        for clock in range(8):
            cutoff_clock=F.one_hot(torch.tensor([clock]),8).float()
            reference=model.forward_heads(windows,cutoff_clock)
            torch.testing.assert_close(grouped['prediction'][:,clock],reference,rtol=1e-5,atol=1e-6)
            gradients=torch.autograd.grad(grouped['prediction'][0,clock,0]+grouped['quantity_delta'][0,clock].sum(),
                                          tuple(windows.values()),retain_graph=clock<7)
            for (frequency,x),gradient in zip(windows.items(),gradients):
                bars=240//int(frequency[3:-1]);previous=config['data']['trading_day_windows'][frequency]['previous_days']
                cutoff=previous*bars+clock*(bars//8)
                self.assertEqual(torch.count_nonzero(gradient[:,:,cutoff:]).item(),0)
                self.assertGreater(gradient[:,:,:cutoff].abs().sum().item(),0)

    def test_primary_priority_projection_hand_cases(self):
        p=torch.nn.Parameter(torch.zeros(2))
        report=combine_gradients([p],[torch.tensor([1.,0.])],[torch.tensor([-1.,2.])],.5,1)
        torch.testing.assert_close(p.grad,torch.tensor([1.,.5]))
        self.assertEqual(float(report['gradient_conflict']),1)
        self.assertAlmostEqual(float(report['projected_auxiliary_dot_primary']),0)
        combine_gradients([p],[torch.tensor([1.,0.])],[torch.tensor([-1.,0.])],.5,1)
        torch.testing.assert_close(p.grad,torch.tensor([1.,0.]))
        combine_gradients([p],[torch.zeros(2)],[torch.ones(2)],.5,1)
        torch.testing.assert_close(p.grad,torch.zeros(2))
        combine_gradients([p],[torch.tensor([1.,1.])],[torch.tensor([1.,0.])],.5,1)
        torch.testing.assert_close(p.grad,torch.tensor([1.5,1.]))

    def test_missing_auxiliary_keeps_primary_and_finite_gradients(self):
        config=small_config();model=MultiFrequencyGRU(config)
        windows,clock=inputs(config)
        targets=trainer.preflight_targets(config,3,torch.device('cpu'))
        targets[...,2:]=float('nan')
        tasks=model.forward_group_tasks(windows,clock)
        main,aux,details=task_objectives(tasks,targets,None,config['training'])
        self.assertEqual(float(aux.detach()),0)
        self.assertEqual(details['quantity_valid_values'],0)
        self.assertEqual(details['quantity_matched_values'],0)
        (main+aux).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError): task_objectives(tasks,targets[...,:2],None,config['training'])

    def test_group_balancing_and_primary_independence(self):
        config=small_config();model=MultiFrequencyGRU(config)
        windows,clock=inputs(config);tasks=model.forward_group_tasks(windows,clock)
        targets=trainer.preflight_targets(config,3,torch.device('cpu'))
        main,aux,details=task_objectives(tasks,targets,None,config['training'])
        changed=targets.clone();changed[...,2:24]+=100
        other,_,_=task_objectives(tasks,changed,None,config['training'])
        torch.testing.assert_close(main,other,rtol=0,atol=0)
        # 三个字段组等权，与2、10、10字段数无关。
        tasks['quantity_delta']=torch.zeros_like(tasks['quantity_delta'],requires_grad=True)
        targets[...,24:46]=0;targets[...,2:24]=1;targets[...,2:4]=2
        _,aux,_=task_objectives(tasks,targets,None,config['training'])
        self.assertAlmostEqual(float(aux.detach()),1.0)

    def test_cpu_bfloat16_two_task_gradients(self):
        config=small_config();model=MultiFrequencyGRU(config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        windows,clock=inputs(config)
        targets=trainer.preflight_targets(config,3,torch.device('cpu'))
        step=trainer.training_step(model,optimizer,windows,clock,targets,config,True)
        self.assertTrue(all(np.isfinite(v) for v in step.values()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreaterEqual(step['projected_auxiliary_dot_primary'],-1e-6)

    def test_exact_optimizer_ema_rng_resume(self):
        config=small_config();torch.manual_seed(15)
        model=MultiFrequencyGRU(config);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        ema=WeightEMA(model,.9,0);windows,clock=inputs(config)
        targets=trainer.preflight_targets(config,3,torch.device('cpu'))
        for _ in range(2):
            trainer.training_step(model,optimizer,windows,clock,targets,config,False);ema.update(1)
        saved=copy.deepcopy((model.state_dict(),optimizer.state_dict(),ema.model.state_dict(),ema.updates,torch.get_rng_state()))
        expected=[]
        for _ in range(2):
            expected.append(trainer.training_step(model,optimizer,windows,clock,targets,config,False));ema.update(1)
        restored=MultiFrequencyGRU(config);other=torch.optim.AdamW(restored.parameters(),lr=.0003);average=WeightEMA(restored,.9,0)
        restored.load_state_dict(saved[0]);other.load_state_dict(saved[1]);average.restore(saved[2],saved[3],saved[3]);torch.set_rng_state(saved[4])
        for reference in expected:
            actual=trainer.training_step(restored,other,windows,clock,targets,config,False);average.update(1)
            self.assertEqual(reference,actual)
        for left,right in [(model,restored),(ema.model,average.model)]:
            for name,value in left.state_dict().items(): torch.testing.assert_close(value,right.state_dict()[name],rtol=0,atol=0)
        for key,state in optimizer.state_dict()['state'].items():
            for name,value in state.items(): torch.testing.assert_close(value,other.state_dict()['state'][key][name],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
