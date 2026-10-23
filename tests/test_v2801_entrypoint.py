"""临时行情上的固定父模型、完整树训练与双集直接推理核验。"""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in ('common','evaluation','solutions/v2801'):
    sys.path.insert(0,str(ROOT/folder))
from artifacts import write_json,read_json,sha256
import v2401_prepare as prepare
import v2401_apply as apply
import v2401_train as parent_train
from v2401_data import ProcessedStore,regular_times
from v2401_model import MultiFrequencyGRU
from v2401_fast import predict_day
from v2801_model import MultiFrequencyNeuralTree
import v2801_run as runner


class EntrypointTests(unittest.TestCase):
    def test_complete_cpu_pipeline_matches_direct_model_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root/'data';raw.mkdir()
            days = pd.to_datetime(['2020-01-02','2020-02-03','2020-03-02'])
            stocks = ['S'+str(i) for i in range(6)]
            times = (570,600,630,660,690,810,840,870)
            keys = pd.DataFrame([(d+pd.Timedelta(minutes=t),s) for d in days for t in times for s in stocks],columns=['date','instrument'])
            keys.to_parquet(raw/'cpt_jyc_2026_instruments.parquet',index=False)
            rng = np.random.default_rng(2801)
            labels = keys.copy();labels['vwap_return']=rng.normal(0,.001,len(labels))
            labels.loc[labels.date.dt.hour.eq(9) & labels.date.dt.month.eq(1),'vwap_return']=np.nan
            labels.to_parquet(raw/'cpt_jyc_2026_vwap.parquet',index=False)
            for frequency in prepare.FREQUENCIES:
                directory=raw/('cpt_jyc_2026_stock_'+frequency);directory.mkdir()
                for day in days:
                    grid=pd.to_datetime(regular_times([day],frequency))
                    frame=pd.DataFrame([(t,s) for t in grid for s in stocks],columns=['date','instrument'])
                    for field in prepare.FIELDS:frame[field]=rng.uniform(1,20,len(frame))
                    frame.to_parquet(directory/(str(day.to_period('M'))+'.parquet'),index=False)
            protocol=read_json(ROOT/'evaluation/protocols/local-v2-ic.json')
            protocol['min_cross_section_size']=3
            protocol['splits']={'train':{'start':'2020-01-01','stop':'2020-02-01'},
                'validation':{'start':'2020-02-01','stop':'2020-03-01'},'test':{'start':'2020-03-01','stop':'2020-04-01'}}
            parent_config=read_json(ROOT/'solutions/v2801/parent-model-config.json')
            parent_config['model'].update(hidden_dim=8,fusion_hidden_dim=16,expert_hidden_dim=8,router_hidden_dim=8,
                book_level_dim=4,book_mixer_hidden_dim=4,non_book_dim=4)
            parent_config['model']['price_memory'].update(width=8,heads=2,distance_hidden_dim=5)
            parent_config['training'].update(cpu_threads=1,stock_day_batch_size=2,batch_size=16)
            parent_config['training']['resident_cache'].update(build_batch_rows=256,evaluation_days=1)
            parent_config['inference'].update(stock_day_batch_size=2,batch_size=16)
            preprocessing=read_json(ROOT/'solutions/v2801/config.json')
            preprocessing['processing'].update(threads=1,batch_rows=256)
            training,holdout=root/'training',root/'holdout'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.prepare(training,raw,preprocessing,protocol)
                apply.apply_holdout(training,holdout,raw)
            parent_run=root/'parent';parent_run.mkdir()
            write_json(parent_run/'config.json',parent_config)
            write_json(parent_run/'protocol.json',protocol)
            torch.manual_seed(42)
            parent=MultiFrequencyGRU(parent_config).eval()
            torch.save({'config':parent_config,'state_dict':parent.state_dict(),'weight_source':'ema','epoch':0},parent_run/'best.pt')
            write_json(parent_run/'manifest.json',{'status':'complete','scope':'synthetic_frozen_parent_fixture'})
            store=ProcessedStore(training,holdout,raw,parent_config,protocol)
            with contextlib.redirect_stdout(io.StringIO()):
                parent_train.data_manifest(store,parent_run)
            preserved={p.relative_to(parent_run).as_posix():sha256(p) for p in parent_run.rglob('*') if p.is_file()}
            config=read_json(ROOT/'solutions/v2801/model-config.json')
            config['backbone_config']=parent_config
            config['data']=copy.deepcopy(parent_config['data'])
            config['model'].update(feature_dim=256,parent_model_sha256=sha256(parent_run/'best.pt'))
            config['training'].update(epochs=2,evaluation_rounds=2,boosting_rounds=4,cpu_threads=1,stock_day_batch_size=2,feature_batch_rows=17)
            config['training']['booster'].update(device='cpu',max_depth=2,max_bin=16,nthread=1,min_child_weight=1.)
            config['inference'].update(stock_day_batch_size=2,feature_batch_rows=13)
            write_json(root/'config.json',config);write_json(root/'protocol.json',protocol)
            args=argparse.Namespace(config=root/'config.json',protocol=root/'protocol.json',run_dir=root/'child',
                parent_run=parent_run,training_run=training,holdout_run=holdout,data_root=raw,
                feature_cache_root=root/'features',evaluation_cache_mode='stream',resident_cache=None,
                resume=False,training_confirmed=True,reproduction_only=True)
            with patch.object(runner,'environment',lambda:{'packages':{}}), contextlib.redirect_stdout(io.StringIO()):
                manifest=runner.execute(args)
            self.assertEqual(manifest['status'],'complete')
            selected=torch.load(args.run_dir/'best.pt',weights_only=True)
            model=MultiFrequencyNeuralTree(config).eval();model.load_state_dict(selected['state_dict'])
            for split in ('validation','test'):
                self.assertEqual(manifest['metrics'][split]['coverage'],1.)
                self.assertEqual(manifest['metrics'][split]['ic_undefined_cross_sections'],0)
                predicted=pd.read_parquet(args.run_dir/'evaluation'/split/'predictions.parquet')
                block=store.load(split,store.months(split)[0])
                direct=predict_day(model,block,torch.device('cpu'),stock_batch_size=2)
                expected=block.keys[['date','instrument']].assign(direct=direct)
                joined=predicted.merge(expected,on=['date','instrument'],validate='one_to_one')
                np.testing.assert_allclose(joined.factor,joined.direct,rtol=1e-5,atol=2e-6)
            self.assertEqual(preserved,{p.relative_to(parent_run).as_posix():sha256(p) for p in parent_run.rglob('*') if p.is_file()})
            self.assertEqual(manifest['selection']['epochs_completed'],2)
            self.assertTrue((args.run_dir/'learning-curves.png').is_file())
            records=read_json(args.run_dir/'feature-caches.json')
            self.assertEqual(set(records),{'train','validation','test'})
            self.assertEqual(read_json(Path(records['train']['path'])/'manifest.json')['finite_primary_rows'],42)
            # 树已选定后只恢复测试阶段，不能再次构造量化矩阵或拟合树。
            before_weight=sha256(args.run_dir/'best.pt')
            before_history=(args.run_dir/'history.csv').read_bytes()
            pending=read_json(args.run_dir/'manifest.json')
            pending.update(status='running',stage='test_evaluation')
            write_json(args.run_dir/'manifest.json',pending)
            args.resume=True
            with patch.object(runner,'environment',lambda:{'packages':{}}), \
                    patch.object(runner,'fit_booster',side_effect=AssertionError('selected trees must not be refitted')), \
                    contextlib.redirect_stdout(io.StringIO()):
                resumed=runner.execute(args)
            self.assertEqual(resumed['status'],'complete')
            self.assertEqual(sha256(args.run_dir/'best.pt'),before_weight)
            self.assertEqual((args.run_dir/'history.csv').read_bytes(),before_history)


if __name__=='__main__':unittest.main()
