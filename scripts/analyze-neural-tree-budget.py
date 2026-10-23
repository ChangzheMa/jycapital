"""只读已完成父版元数据，核对神经树的输入复用与阶段空间预算。"""
import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
for directory in ('common','evaluation','solutions/v2801'):
    sys.path.insert(0,str(ROOT/directory))
from artifacts import read_json,write_json,sha256,now
from v2801_run import validate_training,input_directory
from v2401_data import BaseProcessedStore
from price_memory_inputs import base_configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/v2801-input-cache-compatibility.json')
    args=parser.parse_args()
    path=ROOT/'solutions/v2801/model-config.json'
    config=read_json(path);parent=ROOT/'outputs/runs'/config['parent_run_id']
    protocol=read_json(parent/'protocol.json');source=read_json(parent/'data-manifest.json')
    validate_training(config,protocol)
    if sha256(parent/'best.pt')!=config['model']['parent_model_sha256']:raise ValueError('父权重指纹不同')
    if config['backbone_config']!=read_json(parent/'config.json') or config['data']!=config['backbone_config']['data']:
        raise ValueError('父输入配置不同')
    base=BaseProcessedStore.__new__(BaseProcessedStore)
    base.config=base_configuration(config['backbone_config']);base.protocol=protocol
    store=SimpleNamespace(base=base,training_run=Path(source['preprocessing']['train']['path']))
    prefix=parent/'source/solutions/v2401'
    frozen={}
    for p in (ROOT/'solutions/v2801').glob('v*.py'):
        if not p.name.startswith(('v1601_','v2401_')):continue
        if p.read_bytes()!=(prefix/p.name).read_bytes():raise ValueError('父兼容模块字节改变: '+p.name)
        frozen[p.relative_to(ROOT).as_posix()]=sha256(p)
    group=read_json(parent/'stock-day-training.json')['segments']['global']
    records=[]
    width=config['model']['feature_dim']
    for split in ('train','validation','test'):
        directory=input_directory(store,source,split)
        saved=read_json(directory/'manifest.json')
        rows=saved['catalog']['rows']
        finite=group['eligible_sample_rows'] if split=='train' else read_json(parent/'evaluation'/split/'metrics.json')['evaluated_rows']
        records.append({'split':split,'parent_cache_key':saved['cache_key'],'base_cache_key':saved['base_cache_key'],
            'parent_cache_manifest_sha256':sha256(directory/'manifest.json'),'rows':rows,'finite_primary_rows':finite,
            'feature_array_bytes':rows*width*4,'numeric_metadata_bytes':rows*20,
            'raw_input_resident_array_bytes':saved['resident_array_bytes']})
    train=records[0];bins=config['training']['booster']['max_bin']
    bits=math.ceil(math.log2(width*bins+1))
    result={'checked_at':now(),'solution_id':'v2801','parent_run_id':parent.name,'config_sha256':sha256(path),
        'script_sha256':sha256(__file__),'parent_model_sha256':config['model']['parent_model_sha256'],
        'parent_data_manifest_sha256':sha256(parent/'data-manifest.json'),'base_data_config_identical':True,
        'provided_label_and_primary_transform_unchanged':True,'parent_compatibility_sources':frozen,
        'feature_layout':config['model']['feature_layout'],'feature_dim':width,'splits':records,
        'estimated_feature_and_numeric_metadata_bytes':sum(r['feature_array_bytes']+r['numeric_metadata_bytes'] for r in records),
        'native_train_quantile_host_guard_bytes':train['finite_primary_rows']*width*4+config['training']['feature_batch_rows']*width*8+train['finite_primary_rows']*32,
        'ellpack_data_only_estimate_bytes':math.ceil(train['finite_primary_rows']*width*bits/8),
        'ellpack_assumed_bits_per_element':bits,'gpu_estimate_is_peak':False,
        'raw_input_arrays_released_before_tree_fit':True,'configured_host_memory_reserve_gib':2.,
        'maximum_boosting_rounds':config['training']['boosting_rounds'],'maximum_tree_numeric_parameters':config['training']['boosting_rounds']*(2**(config['training']['booster']['max_depth']+1)-1),
        'formal_training_performed':False,'input_numeric_arrays_loaded':False,
        'limitations':['表示缓存新增空间还需加评测keys.parquet、清单和运行输出；不会复制父完整行情数组。',
            'ELLPACK数字仅为量化数据负载估计，不包含转换峰值、工作区、梯度和原生缓存；完整GPU矩阵内存尚未实测。',
            '主机量化构建按批读取，完整浮点表示以mmap保存；不能把磁盘映射长度当作始终常驻内存。',
            '缓存身份绑定父权重、输入和表示源码，不含树超参数或版本号；后续相同表示须复用这些冻结兼容模块。']}
    write_json(args.output,result)
    print({k:result[k] for k in ('feature_dim','estimated_feature_and_numeric_metadata_bytes','native_train_quantile_host_guard_bytes','ellpack_data_only_estimate_bytes')})


if __name__=='__main__':main()
