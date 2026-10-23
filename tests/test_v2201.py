"""跨日同阶段编码的双轴因果边界、共享前缀及恢复检查。"""
import copy
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for version in ('v1601','v2201'):sys.path.insert(0,str(ROOT/'solutions'/version))
import v2201_train as trainer
from v2201_model import MultiFrequencyGRU
from v1601_model import MultiFrequencyGRU as ParentModel
from v2201_ema import WeightEMA

CONFIG=json.loads((ROOT/'solutions/v2201/model-config.json').read_text(encoding='utf-8'))
CONFIGS={'v2201':CONFIG};TEST_VERSIONS=('v2201',);PARENTS={'v2201':'v1601'}
PROTOCOL=json.loads((ROOT/'evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))
torch.set_num_threads(1)


def small_config(version='v2201',resident=False):
    config=copy.deepcopy(CONFIG)
    config['model'].update(hidden_dim=8,fusion_hidden_dim=16,expert_hidden_dim=8,router_hidden_dim=8,
                           book_level_dim=4,book_mixer_hidden_dim=4,non_book_dim=4)
    config['training'].update(epochs=2,cpu_threads=1,batch_size=16,stock_day_batch_size=2,checkpoint_batches=1)
    config['inference'].update(batch_size=16,stock_day_batch_size=2)
    config['training']['resident_cache'].update(memory_reserve_gib=0,build_batch_rows=256,evaluation_days=1)
    if not resident:
        config['training']['batching']='monthly_shuffled_stock_days'
        config['data']['input_precision']='float32'
    return config


def inputs(config,count=3):
    generator=torch.Generator().manual_seed(55)
    windows={f:torch.randn(count,75,n,generator=generator) for f,n in config['data']['window_bars'].items()}
    for value in windows.values():value[:,72:]=1
    return windows,F.one_hot(torch.full((count,),7),8).float()


class PhaseHistoryTests(unittest.TestCase):
    def test_default_size_and_contract(self):
        trainer.validate_training(CONFIG,PROTOCOL)
        self.assertEqual(sum(p.numel() for p in MultiFrequencyGRU(CONFIG).parameters()),4528944)
        self.assertEqual(CONFIG['training']['resident_cache']['memory_reserve_gib'],2)

    def test_nonzero_phase_all_clock_causality_and_shared_reference(self):
        config=small_config();torch.manual_seed(21);model=MultiFrequencyGRU(config).eval()
        with torch.no_grad():
            for encoder in model.encoders.values():encoder.phase_embedding.normal_(0,.2)
        windows,latest=inputs(config,2);windows={f:x.requires_grad_() for f,x in windows.items()}
        grouped=model.forward_group(windows,latest)
        for clock in range(8):
            clock_values=F.one_hot(torch.full((2,),clock),8).float()
            reference=model.forward_heads(windows,clock_values)
            torch.testing.assert_close(grouped[:,clock],reference,rtol=1e-5,atol=1e-6)
            gradients=torch.autograd.grad(grouped[:,clock].sum(),tuple(windows.values()),retain_graph=clock<7)
            for (frequency,_),gradient in zip(windows.items(),gradients):
                bars=240//int(frequency[3:-1]);previous=config['data']['trading_day_windows'][frequency]['previous_days']
                cutoff=previous*bars+clock*(bars//8)
                self.assertEqual(torch.count_nonzero(gradient[:,:,cutoff:]).item(),0)
                self.assertGreater(gradient[:,:,:cutoff].abs().sum().item(),0)
        partial=model.forward_group(windows,F.one_hot(torch.full((2,),3),8).float())
        torch.testing.assert_close(partial[:,:4],grouped[:,:4],rtol=1e-5,atol=1e-6)

    def test_same_phase_history_matches_independent_phase_sequences(self):
        model=MultiFrequencyGRU(small_config()).eval()
        for frequency in ('bar5m','bar30m'):
            encoder=model.encoders[frequency]
            h=torch.randn(2,encoder.days*encoder.daily_bars,8)
            actual=encoder.aligned_history(h)
            view=h.reshape(2,encoder.days,encoder.daily_bars,8)
            expected=torch.stack([encoder.interday(view[:,:,phase])[0] for phase in range(encoder.daily_bars)],dim=2)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=3e-7)
            changed=view.clone();changed[:,-1,-1]+=100
            other=encoder.aligned_history(changed.reshape_as(h))
            torch.testing.assert_close(actual[:,:-1],other[:,:-1],rtol=0,atol=0)
            torch.testing.assert_close(actual[:,:,:-1],other[:,:,:-1],rtol=0,atol=0)

    def test_previous_late_phase_carries_into_current_prefix(self):
        model=MultiFrequencyGRU(small_config()).eval()
        for frequency in ('bar5m','bar30m'):
            encoder=model.encoders[frequency]
            h=torch.randn(1,encoder.days*encoder.daily_bars,8,requires_grad=True)
            states=encoder.phase_states(h)
            gradient=torch.autograd.grad(states[:,encoder.daily_bars].sum(),h)[0]
            previous_last=(encoder.days-1)*encoder.daily_bars-1
            current_first=previous_last+1
            self.assertGreater(gradient[:,previous_last].abs().sum().item(),0)
            self.assertEqual(torch.count_nonzero(gradient[:,current_first+1:]).item(),0)

    def test_capacity_matched_common_initialization_and_minute_path(self):
        config=small_config();parent_config=copy.deepcopy(config);parent_config['solution_id']='v1601'
        for key in ('temporal_axis_order','phase_embedding','phase_embedding_initial'):parent_config['model'].pop(key)
        torch.manual_seed(42);new=MultiFrequencyGRU(config).eval()
        torch.manual_seed(42);old=ParentModel(parent_config).eval()
        for name,value in old.state_dict().items():torch.testing.assert_close(new.state_dict()[name],value,rtol=0,atol=0)
        extra=set(new.state_dict())-set(old.state_dict())
        self.assertEqual(extra,{f'encoders.{f}.phase_embedding' for f in config['data']['window_bars']})
        windows,_=inputs(config,2);lengths=torch.tensor([240,450])
        torch.testing.assert_close(new.encoders['bar1m'](windows['bar1m'],lengths),old.encoders['bar1m'](windows['bar1m'],lengths),rtol=0,atol=0)

    def test_actual_objective_and_bfloat16_gradients(self):
        config=small_config();model=MultiFrequencyGRU(config)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003);windows,clock=inputs(config)
        targets=torch.linspace(-2,2,48,dtype=torch.float64).reshape(3,8,2)
        step=trainer.training_step(model,optimizer,windows,clock,targets,config,True)
        self.assertTrue(np.isfinite(step['loss']))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_exact_raw_ema_optimizer_and_rng_resume(self):
        config=small_config();torch.manual_seed(14)
        model=MultiFrequencyGRU(config);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003);ema=WeightEMA(model,.9,0)
        windows,clock=inputs(config);targets=torch.linspace(-2,2,48,dtype=torch.float64).reshape(3,8,2)
        for _ in range(2):trainer.training_step(model,optimizer,windows,clock,targets,config,False);ema.update(1)
        saved=copy.deepcopy((model.state_dict(),optimizer.state_dict(),ema.model.state_dict(),ema.updates,torch.get_rng_state()))
        expected=[]
        for _ in range(2):expected.append(trainer.training_step(model,optimizer,windows,clock,targets,config,False));ema.update(1)
        other=MultiFrequencyGRU(config);opt=torch.optim.AdamW(other.parameters(),lr=.0003);average=WeightEMA(other,.9,0)
        other.load_state_dict(saved[0]);opt.load_state_dict(saved[1]);average.restore(saved[2],saved[3],saved[3]);torch.set_rng_state(saved[4])
        for reference in expected:
            self.assertEqual(trainer.training_step(other,opt,windows,clock,targets,config,False),reference);average.update(1)
        for left,right in [(model,other),(ema.model,average.model)]:
            for name,value in left.state_dict().items():torch.testing.assert_close(value,right.state_dict()[name],rtol=0,atol=0)
        for key,state in optimizer.state_dict()['state'].items():
            for name,value in state.items():torch.testing.assert_close(value,opt.state_dict()['state'][key][name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
