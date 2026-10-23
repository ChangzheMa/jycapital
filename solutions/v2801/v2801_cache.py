"""分阶段缓存冻结历史表示；树拟合阶段无需保留原始常驻行情。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
import numpy as np

from artifacts import read_json, write_json, sha256, now
from coordinate_resident import _remove_own_staging
from v2401_grouped import StockDayIndex
from v2401_resident import ResidentBatches, require_memory
from v2401_text_artifact import state_digest
import v2801_features

FORMAT = 'jycapital-frozen-neural-features-v1'


class ZeroTargets:
    def __getitem__(self, indices):
        return np.zeros(len(indices), np.float32)


def array_record(path):
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        return {'shape':list(value.shape),'dtype':value.dtype.str,'array_bytes':value.nbytes,
                'bytes':path.stat().st_size,'sha256':sha256(path)}
    finally:
        value._mmap.close()


def cache_identity(config, source_manifest, block_manifest, split):
    identity = {'format':FORMAT,'split':split,'parent_model_sha256':config['model']['parent_model_sha256'],
        'feature_layout':config['model']['feature_layout'],'features':config['model']['feature_dim'],
        'parent_data_manifest':source_manifest,'parent_input_cache_key':block_manifest['cache_key'],
        'parent_input_files':block_manifest['files'],'parent_base_files':block_manifest.get('base_files'),
        'feature_source_sha256':sha256(v2801_features.__file__),
        'builder_source_sha256':sha256(__file__)}
    content = json.dumps(identity,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    return hashlib.sha256(content).hexdigest(), identity


def date_weights(dates, primary, policy, interval):
    dates = np.asarray(dates).astype('datetime64[ns]').astype('datetime64[D]')
    start, stop = np.datetime64(interval['start'],'D'), np.datetime64(interval['stop'],'D')
    if np.isnat(dates).any() or (dates < start).any() or (dates >= stop).any():
        raise ValueError('训练权重的日期越过固定训练区间')
    raw = policy['floor'] + (1-policy['floor']) * np.exp2(-(stop-dates).astype(np.float64)/policy['half_life_days'])
    finite = np.isfinite(primary)
    if not finite.any():
        raise ValueError('树训练没有有效主标签')
    normalizer = float(raw[finite].mean())
    return (raw/normalizer).astype(np.float32), {'policy':dict(policy),'interval':dict(interval),
        'normalization':'mean_weight_one_over_all_finite_training_primary_rows','raw_mean':normalizer,
        'finite_primary_rows':int(finite.sum())}


def build_features(directory, identity, block, encoder, config, protocol, split, *, keys=None, report=None):
    directory = Path(directory)
    if directory.exists():
        saved = read_json(directory / 'manifest.json')
        if saved.get('status') != 'complete' or saved.get('identity') != identity:
            raise ValueError('学习表示缓存未完成或身份改变')
        return saved
    directory.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=directory.name+'.building-',dir=directory.parent))
    matrix = None
    before = state_digest(encoder.state_dict())
    try:
        index = StockDayIndex(block)
        rows, width = len(block.targets), encoder.feature_dim
        if width != config['model']['feature_dim']:
            raise ValueError('实际学习表示宽度与配置不同')
        batch_size = config['training']['stock_day_batch_size']
        if shutil.disk_usage(staging).free < rows*(width*4+24)+2**30:
            raise OSError('磁盘空间不足以保存本次学习表示及写入余量')
        require_memory(batch_size * 8 * width * 4 * 3 + rows, 2.)
        matrix = np.lib.format.open_memmap(staging/'features.npy',mode='w+',dtype=np.float32,shape=(rows,width))
        seen = np.zeros(rows,bool)
        device = next(encoder.parameters()).device
        encoder.eval()
        iterator = ResidentBatches(block,index.representatives,ZeroTargets(),0,batch_size,device)
        with torch.inference_mode():
            for offset, selected, windows, clock, _ in iterator:
                features, _ = encoder.group_features({f:x.contiguous() for f,x in windows.items()},clock)
                data = features.cpu().numpy()
                positions = index.samples[offset:offset+len(selected)]
                valid = positions >= 0
                row_ids = positions[valid]
                if seen[row_ids].any() or not np.isfinite(data[valid]).all():
                    raise ValueError('学习表示重复、缺失或含非有限数值')
                matrix[row_ids] = data[valid]
                seen[row_ids] = True
                if offset // batch_size % 128 == 0:
                    matrix.flush()
                    if report:report('neural_features_progress',split=split,completed_groups=int(offset+len(selected)),groups=len(index.samples))
        if not seen.all():
            raise ValueError('学习表示没有覆盖完整原始请求键')
        matrix.flush()
        matrix._mmap.close()
        matrix = None
        if state_digest(encoder.state_dict()) != before:
            raise ValueError('冻结历史编码器的参数在提取期间改变')
        primary = np.asarray(block.targets,dtype=np.float64)
        np.save(staging/'primary.npy',primary,allow_pickle=False)
        dates = np.asarray(block.arrays['days'])[block.day].astype('datetime64[ns]')
        np.save(staging/'date.npy',dates.astype(np.int64),allow_pickle=False)
        if split == 'train':
            weights, weight_record = date_weights(dates,primary,config['training']['temporal_weighting'],protocol['splits']['train'])
        else:
            weights, weight_record = np.ones(rows,np.float32), None
        np.save(staging/'weights.npy',weights,allow_pickle=False)
        key_record = None
        if keys is not None:
            if len(keys) != rows or keys.duplicated(['date','instrument']).any():
                raise ValueError('学习表示与评测键不匹配')
            keys.to_parquet(staging/'keys.parquet',index=False)
            key_record = {'bytes':(staging/'keys.parquet').stat().st_size,'sha256':sha256(staging/'keys.parquet')}
        saved = {'format':FORMAT,'status':'complete','created_at':now(),'identity':identity,
            'rows':rows,'feature_dim':width,'finite_primary_rows':int(np.isfinite(primary).sum()),
            'encoder_tensor_sha256':before,'encoder_unchanged':True,'temporal_weighting':weight_record,
            'files':{p.name:array_record(p) for p in sorted(staging.glob('*.npy'))},'keys':key_record,
            'primary_targets_modified':False,'prediction_margin_column':width-1}
        write_json(staging/'manifest.json',saved)
        os.replace(staging,directory)
        return saved
    except BaseException:
        if matrix is not None:matrix._mmap.close()
        if staging.exists():_remove_own_staging(staging,directory)
        raise


class FeatureCache:
    def __init__(self, directory, identity=None):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory/'manifest.json')
        self.arrays = {}
        saved = self.manifest
        if saved.get('format') != FORMAT or saved.get('status') != 'complete':
            raise ValueError('学习表示缓存未完成')
        if identity is not None and saved['identity'] != identity:
            raise ValueError('学习表示来源改变')
        try:
            for name, record in saved['files'].items():
                if name not in ('features.npy','primary.npy','date.npy','weights.npy'):
                    raise ValueError('学习表示缓存含未声明数组')
                path = self.directory/name
                if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                    raise ValueError('学习表示数组指纹不一致')
                value = np.load(path,mmap_mode='r',allow_pickle=False)
                self.arrays[name[:-4]] = value
                if list(value.shape) != record['shape'] or value.dtype.str != record['dtype']:
                    raise ValueError('学习表示数组布局改变')
            if set(self.arrays) != {'features','primary','date','weights'}:
                raise ValueError('学习表示缓存缺少数组')
            if self.arrays['features'].shape != (saved['rows'],saved['feature_dim']):
                raise ValueError('学习表示维度与清单不同')
            for key in ('primary','date','weights'):
                if self.arrays[key].shape != (saved['rows'],):
                    raise ValueError('学习表示辅助数组行数不符')
            if saved['keys'] and sha256(self.directory/'keys.parquet') != saved['keys']['sha256']:
                raise ValueError('评测键或原始标签改变')
        except BaseException:
            self.close()
            raise

    def close(self):
        for value in self.arrays.values():value._mmap.close()
        self.arrays.clear()
