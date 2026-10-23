"""原始状态选择、成员监督、共享投影和精确恢复的合成检查。"""
import copy
import json
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for folder in ('scripts','solutions/v2301'):sys.path.insert(0,str(ROOT/folder))
import v2301_train as trainer
from v2301_model import MultiFrequencySnapshotMLP, EnsembleLinear
from v2301_objective import combined_objective
from v2301_ema import WeightEMA
from v2301_text_artifact import encode_state, decode_state, state_digest, FORMAT
from assess import result_filename

CONFIG=json.loads((ROOT/'solutions/v2301/model-config.json').read_text(encoding='utf-8'))
CONFIGS={'v2301':CONFIG};TEST_VERSIONS=('v2301',);PARENTS={'v2301':'v1601'}
PROTOCOL=json.loads((ROOT/'evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))
torch.set_num_threads(1)


def small_config(version='v2301',resident=False):
    config=copy.deepcopy(CONFIG);config['model'].update(width=12,ensemble_size=3)
    config['training'].update(epochs=2,cpu_threads=1,batch_size=16,stock_day_batch_size=2,checkpoint_batches=1)
    config['inference'].update(batch_size=16,stock_day_batch_size=2)
    config['training']['resident_cache'].update(memory_reserve_gib=0,build_batch_rows=256,evaluation_days=1)
    if not resident:
        config['training']['batching']='monthly_shuffled_stock_days';config['data']['input_precision']='float32'
    return config


def inputs(config,count=3):
    generator=torch.Generator().manual_seed(55)
    windows={f:torch.randn(count,75,n,generator=generator) for f,n in config['data']['window_bars'].items()}
    for value in windows.values():value[:,72:]=1
    return windows,F.one_hot(torch.full((count,),7),8).float()


class DirectStateTests(unittest.TestCase):
    def test_default_contract_and_family_registration(self):
        trainer.validate_training(CONFIG,PROTOCOL);model=MultiFrequencySnapshotMLP(CONFIG)
        self.assertEqual(sum(p.numel() for p in model.parameters()),2051424)
        self.assertEqual(model.input_dim,2858)
        self.assertEqual(CONFIG['training']['resident_cache']['memory_reserve_gib'],2)
        self.assertEqual(result_filename('v2301','snapshot_mlp'),'v2301-snapshot-mlp.json')
        self.assertEqual(result_filename('v1601','hierarchical_gru'),'v1601-gru.json')
        with self.assertRaises(ValueError):result_filename('../bad','snapshot_mlp')

    def test_exact_raw_positions_and_completed_day_anchors(self):
        cfg=small_config();model=MultiFrequencySnapshotMLP(cfg).eval();windows,clock=inputs(cfg,1)
        for frequency,value in windows.items():
            value[:]=torch.arange(value.shape[-1])[None,None,:]+1000*torch.arange(75)[None,:,None]
        recent,anchors,clocks=model.group_features(windows,clock)
        expected=[]
        for c in range(8):
            parts=[]
            for f,x in windows.items():
                bars=240//int(f[3:-1]);prior=cfg['data']['trading_day_windows'][f]['previous_days']
                end=prior*bars+c*(bars//8);count=cfg['model']['snapshot_bars'][f]
                parts.append(x[0,:,end-count:end].T.flatten())
            expected.append(torch.cat(parts))
        torch.testing.assert_close(recent[0],torch.stack(expected),rtol=0,atol=0)
        expected_anchors=torch.cat([windows['bar30m'][0,:,day*8+7] for day in range(20)])
        torch.testing.assert_close(anchors[0],expected_anchors,rtol=0,atol=0)
        self.assertEqual(int(model.anchor_indices.max()),159)

    def test_shared_linear_matches_explicit_member_matrices(self):
        torch.manual_seed(1);layer=EnsembleLinear(6,4,3,first=True).double()
        with torch.no_grad():layer.r.uniform_(.3,1.2);layer.s.uniform_(.4,1.3)
        x=torch.randn(5,3,6,dtype=torch.float64)
        expected=torch.stack([F.linear(x[:,i],layer.linear.weight*layer.s[i,:,None]*layer.r[i,None,:],layer.bias[i]) for i in range(3)],dim=1)
        torch.testing.assert_close(layer(x),expected,rtol=1e-12,atol=1e-12)
        first=EnsembleLinear(6,4,3,first=True)
        self.assertTrue(torch.all((first.r==1)|(first.r==-1)))
        later=EnsembleLinear(4,4,3)
        self.assertTrue(torch.all(later.r==1)&torch.all(later.s==1))

    def test_shared_parts_all_clock_reference_and_future_gradients(self):
        cfg=small_config();torch.manual_seed(12);model=MultiFrequencySnapshotMLP(cfg).eval()
        with torch.no_grad():
            for layer in model.layers:layer.r.normal_(1,.3);layer.s.normal_(1,.2)
        windows,clock=inputs(cfg,2);windows={f:x.requires_grad_() for f,x in windows.items()}
        grouped=model.forward_group_members(windows,clock)
        for c in range(8):
            reference=model.forward_member_heads(windows,F.one_hot(torch.full((2,),c),8).float())
            torch.testing.assert_close(grouped[:,c],reference,rtol=1e-5,atol=2e-6)
            gradients=torch.autograd.grad(grouped[:,c].sum(),tuple(windows.values()),retain_graph=c<7)
            for (frequency,_),gradient in zip(windows.items(),gradients):
                bars=240//int(frequency[3:-1]);prior=cfg['data']['trading_day_windows'][frequency]['previous_days']
                cutoff=prior*bars+c*(bars//8)
                self.assertEqual(torch.count_nonzero(gradient[:,:,cutoff:]).item(),0)
            self.assertEqual(torch.count_nonzero(gradients[0][:,:,:200]).item(),0)
        partial=model.forward_group_members(windows,F.one_hot(torch.full((2,),3),8).float())
        torch.testing.assert_close(partial[:,:4],grouped[:,:4],rtol=1e-5,atol=2e-6)
        split=torch.cat([model.forward_group_members({f:x[i:i+1] for f,x in windows.items()},clock[i:i+1]) for i in range(2)])
        torch.testing.assert_close(split,grouped,rtol=1e-5,atol=2e-6)

    def test_individual_losses_do_not_collapse_to_mean_prediction_loss(self):
        cfg=small_config();targets=torch.zeros(2,8,2,dtype=torch.float64)
        values=torch.zeros(2,8,2,2,dtype=torch.float64);values[:,:,0,0]=-1;values[:,:,1,0]=1
        values.requires_grad_();loss,details=combined_objective(values,targets,None,cfg['training'])
        self.assertAlmostEqual(float(loss.detach()),.5)
        self.assertEqual(float(details['ensemble_huber_loss']),0)
        self.assertAlmostEqual(float(details['member_disagreement_rms']),1)
        loss.backward();self.assertGreater(values.grad.abs().sum().item(),0)
        targets[:,0]=float('nan')
        loss,details=combined_objective(values,targets,None,cfg['training'])
        self.assertEqual(details['rows'],14)
        self.assertAlmostEqual(float(loss.detach()),.5)

    def test_empty_history_and_cpu_bfloat16_actual_objective(self):
        cfg=small_config();model=MultiFrequencySnapshotMLP(cfg);windows,clock=inputs(cfg)
        empty={f:torch.zeros_like(x) for f,x in windows.items()}
        self.assertTrue(torch.isfinite(model.eval().forward_group(empty,clock)).all())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        targets=torch.linspace(-2,2,48,dtype=torch.float64).reshape(3,8,2)
        step=trainer.training_step(model,optimizer,windows,clock,targets,cfg,True)
        self.assertTrue(all(np.isfinite(x) for x in step.values()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_text_roundtrip_preserves_member_tensors_and_predictions(self):
        cfg=small_config();model=MultiFrequencySnapshotMLP(cfg).eval()
        bundle=json.loads(json.dumps({'format':FORMAT,'state_dict':encode_state(model.state_dict())}))
        decoded=decode_state(bundle)
        self.assertEqual(state_digest(model.state_dict()),state_digest(decoded))
        restored=MultiFrequencySnapshotMLP(cfg).eval();restored.load_state_dict(decoded,strict=True)
        windows,clock=inputs(cfg)
        torch.testing.assert_close(model.forward_group(windows,clock),restored.forward_group(windows,clock),rtol=0,atol=0)

    def test_exact_raw_ema_optimizer_rng_resume(self):
        cfg=small_config();torch.manual_seed(15);model=MultiFrequencySnapshotMLP(cfg)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003);ema=WeightEMA(model,.9,0)
        windows,clock=inputs(cfg);targets=torch.linspace(-2,2,48,dtype=torch.float64).reshape(3,8,2)
        for _ in range(2):trainer.training_step(model,optimizer,windows,clock,targets,cfg,False);ema.update(1)
        saved=copy.deepcopy((model.state_dict(),optimizer.state_dict(),ema.model.state_dict(),ema.updates,torch.get_rng_state()))
        expected=[]
        for _ in range(2):expected.append(trainer.training_step(model,optimizer,windows,clock,targets,cfg,False));ema.update(1)
        other=MultiFrequencySnapshotMLP(cfg);opt=torch.optim.AdamW(other.parameters(),lr=.0003);average=WeightEMA(other,.9,0)
        other.load_state_dict(saved[0]);opt.load_state_dict(saved[1]);average.restore(saved[2],saved[3],saved[3]);torch.set_rng_state(saved[4])
        for reference in expected:
            self.assertEqual(trainer.training_step(other,opt,windows,clock,targets,cfg,False),reference);average.update(1)
        for left,right in [(model,other),(ema.model,average.model)]:
            for name,value in left.state_dict().items():torch.testing.assert_close(value,right.state_dict()[name],rtol=0,atol=0)
        for key,state in optimizer.state_dict()['state'].items():
            for name,value in state.items():torch.testing.assert_close(value,opt.state_dict()['state'][key][name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
