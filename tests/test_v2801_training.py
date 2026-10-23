"""固定表示的分批量化、验证选择和树迭代恢复；全部使用临时合成数据。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for folder in ('common','evaluation','solutions/v2801'):
    sys.path.insert(0,str(ROOT/folder))
from artifacts import write_json
from v2801_cache import FORMAT as CACHE_FORMAT, FeatureCache, array_record, date_weights
from v2801_boosting import TrainingIterator, make_matrix, fit_booster


def config():
    value = json.loads((ROOT/'solutions/v2801/model-config.json').read_text(encoding='utf-8'))
    value['model']['feature_dim'] = 5
    value['training'].update(epochs=4,evaluation_rounds=2,boosting_rounds=8,cpu_threads=1,feature_batch_rows=17)
    value['training']['booster'].update(device='cpu',max_depth=2,max_bin=16,nthread=1,eta=.2,min_child_weight=1.,**{'lambda':1.})
    value['inference']['feature_batch_rows'] = 13
    return value


def fixture(directory, split, seed):
    directory.mkdir()
    rng = np.random.default_rng(seed)
    times = pd.to_datetime(['2020-01-02 10:00','2020-01-02 10:30']) if split=='train' else pd.to_datetime(['2024-01-02 10:00','2024-01-02 10:30'])
    keys = pd.DataFrame([(t,'S'+str(i)) for t in times for i in range(32)],columns=['date','instrument'])
    features = rng.normal(size=(len(keys),5)).astype(np.float32)
    response = .5*features[:,0]+(features[:,1]>.2).astype(np.float32)
    primary = np.concatenate([(x-x.mean())/x.std() for x in np.split(response,2)]).astype(np.float64)
    features[:,-1] = .05*features[:,-1]+.1*features[:,0]
    keys['vwap_return'] = .001*response
    if split=='train':primary[0]=np.nan
    arrays = {'features':features,'primary':primary,'weights':np.ones(len(keys),np.float32),
              'date':keys.date.dt.normalize().to_numpy(dtype='datetime64[ns]').astype(np.int64)}
    for name,values in arrays.items():np.save(directory/(name+'.npy'),values,allow_pickle=False)
    keys.to_parquet(directory/'keys.parquet',index=False)
    from artifacts import sha256
    write_json(directory/'manifest.json',{'format':CACHE_FORMAT,'status':'complete','rows':len(keys),
        'feature_dim':5,'finite_primary_rows':int(np.isfinite(primary).sum()),'identity':{'synthetic':seed},
        'files':{p.name:array_record(p) for p in directory.glob('*.npy')},
        'keys':{'sha256':sha256(directory/'keys.parquet')}})
    return FeatureCache(directory)


class TrainingTests(unittest.TestCase):
    def test_finite_filter_and_fixed_date_weight_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = fixture(Path(temporary)/'train','train',42)
            try:
                matrix = make_matrix(cache,config())
                self.assertEqual(matrix.num_row(),63)
                self.assertEqual(matrix.num_col(),5)
                valid = np.isfinite(cache.arrays['primary'])
                np.testing.assert_array_equal(matrix.get_label(),cache.arrays['primary'][valid].astype(np.float32))
                np.testing.assert_array_equal(matrix.get_base_margin(),cache.arrays['features'][valid,-1])
                dates = np.array(['2020-01-02','2023-12-29'],dtype='datetime64[D]')
                weights,record = date_weights(dates,np.ones(2),config()['training']['temporal_weighting'],
                                               {'start':'2020-01-01','stop':'2024-01-01'})
                self.assertAlmostEqual(float(weights.mean()),1.,places=6)
                self.assertGreater(weights[1],weights[0])
                self.assertEqual(record['finite_primary_rows'],2)
            finally:cache.close()

    def test_interrupted_tree_updates_resume_to_identical_model_and_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train,valid = fixture(root/'train','train',42),fixture(root/'validation','validation',43)
            protocol = json.loads((ROOT/'evaluation/protocols/local-v2-ic.json').read_text(encoding='utf-8'))
            protocol['min_cross_section_size'] = 3
            cfg = config()
            encoder_state = {'synthetic_weight':torch.ones(2)}
            paths = [root/'continuous',root/'interrupted']
            for path in paths:path.mkdir()
            try:
                left,left_selection = fit_booster(train,valid,cfg,{},encoder_state,protocol,paths[0],'cpu')
                def interrupt(iteration):
                    if iteration==3:raise KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    fit_booster(train,valid,cfg,{},encoder_state,protocol,paths[1],'cpu',after_update=interrupt)
                saved = torch.load(paths[1]/'last.pt',weights_only=True)
                self.assertEqual(saved['boosting_rounds'],2)
                right,right_selection = fit_booster(train,valid,cfg,{},encoder_state,protocol,paths[1],'cpu',resume=True)
                self.assertEqual(left_selection['best_epoch'],right_selection['best_epoch'])
                self.assertEqual(left_selection['best_validation_ic'],right_selection['best_validation_ic'])
                for name,value in left['state_dict'].items():torch.testing.assert_close(value,right['state_dict'][name],rtol=0,atol=0)
                lhs,rhs = [pd.read_csv(path/'history.csv') for path in paths]
                cols = ['epoch','boosting_rounds','train_loss','validation_loss','validation_ic_mean','best_epoch','improved']
                pd.testing.assert_frame_equal(lhs[cols],rhs[cols],check_exact=True)
            finally:
                train.close();valid.close()


if __name__=='__main__':unittest.main()
