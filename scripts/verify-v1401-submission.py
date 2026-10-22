"""核验双组件文本权重及共享 fast 入口，只使用小型合成行情。"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'common', ROOT / 'solutions/v1002', ROOT / 'solutions/v1101'):
    sys.path.insert(0, str(directory))
from artifacts import read_json, write_json, sha256, now
from v1002_text_artifact import decode_state as decode_temporal
from v1101_text_artifact import decode_state as decode_structured

ISOLATED_CHECK = r'''
import json, pathlib, sys, types, os, site, re
sys.path.append(site.getusersitepackages())
import torch
import numpy as np
import pandas as pd
torch.set_num_threads(4)
package = pathlib.Path.cwd()
assert set(p.name for p in package.iterdir()) == {'v1401-submission.ipynb','v1002-model.json','v1101-model.json'}
repository = pathlib.Path(os.environ['SUBMISSION_REPOSITORY_ROOT']).resolve()
environment = pathlib.Path(sys.prefix).resolve()
for entry in sys.path:
    location = pathlib.Path(entry).resolve()
    assert location != repository
    assert repository not in location.parents or environment in location.parents or location == environment
guard = types.ModuleType('dai')
def forbidden_query(*args, **kwargs):
    raise AssertionError('import must not query data')
guard.query = forbidden_query
sys.modules['dai'] = guard
book = json.loads((package/'v1401-submission.ipynb').read_text(encoding='utf-8'))
scope = {'__name__':'submission_import','__file__':str(package/'submission.py')}
for cell in book['cells']:
    if cell['cell_type']=='code':
        exec(compile(''.join(cell['source']),'<notebook>','exec'),scope)
assert scope['_RUNTIME'] is None
bundles, modules = scope['_runtime']()
model = modules['v1401_model'].FrozenEnsemble([b['config'] for b in bundles]).eval()
for component, bundle in zip(model.components,bundles):
    component.load_state_dict(modules[bundle['config']['solution_id']+'_text_artifact'].decode_state(bundle))
data, combine = modules['v1002_data'], modules['v1401_combine']
days = pd.bdate_range('2024-01-02',periods=22)
stocks = pd.Index(['S0','S1','S2','S3','S4'])
rng = np.random.default_rng(1401)
grids = {}
for frequency in data.FREQUENCIES:
    times = data.regular_times(days,frequency)
    values = rng.normal(size=(len(times),len(stocks),72)).astype(np.float32)
    values[3,1] = np.nan
    grids[frequency] = times,values
minutes = (570,600,630,660,690,810,840,870)
keys = pd.DataFrame([(days[-1]+pd.Timedelta(minutes=m),s) for m in minutes for s in stocks],columns=['date','instrument'])
keys = keys.drop(index=[1,7,18]).sample(frac=1,random_state=5).reset_index(drop=True)
block = data.WindowBlock(keys,grids,stocks,bundles[0]['config'],days.to_numpy())
actual = modules['v1401_fast'].predict_day(model,block,torch.device('cpu'),stock_batch_size=2)
windows,clock = block.batch(np.arange(len(keys)))
with torch.inference_mode():
    raw = model.component_scores({f:torch.from_numpy(x) for f,x in windows.items()},torch.from_numpy(clock)).numpy()
reference = combine.combine_scores(keys,raw).factor.to_numpy()
np.testing.assert_allclose(actual,reference,atol=2e-5,rtol=1e-4)
assert actual.dtype == np.dtype('float64') and np.isfinite(actual).all()

class FakeDai:
    def __init__(self,tables):
        self.tables,self.queries=tables,[]
    def query(self,sql,filters=None):
        self.queries.append(sql)
        table=re.search(r'FROM ([A-Za-z0-9_]+)',sql).group(1)
        frame=self.tables[table].copy()
        for operator,match in (('<',re.search(r"date < '([^']+)'",sql)),('>=',re.search(r"date >= '([^']+)'",sql))):
            if match:
                bound=pd.Timestamp(match.group(1))
                frame=frame.loc[frame.date<bound] if operator=='<' else frame.loc[frame.date>=bound]
        included=re.search(r'instrument IN \(([^)]+)\)',sql)
        if included:
            frame=frame.loc[frame.instrument.isin(re.findall(r"'([^']+)'",included.group(1)))]
        if 'max(date)' in sql:
            frame=pd.DataFrame({'date':[frame.date.max()]})
        elif 'CAST(date AS DATE)' in sql:
            frame=pd.DataFrame({'date':sorted(frame.date.dt.normalize().unique(),reverse=True)})
            limit=re.search(r'LIMIT (\d+)',sql)
            frame=frame.head(int(limit.group(1)))
        elif 'SELECT DISTINCT instrument' in sql:
            frame=frame[['instrument']].drop_duplicates()
        else:
            frame=frame[sql.split('SELECT ',1)[1].split(' FROM',1)[0].split(', ')]
        return type('Query',(),{'df':lambda _:frame})()

tiny_days=pd.bdate_range('2024-01-02',periods=3)
tiny_stocks=['NEW0','NEW1','NEW2']
tables={'cpt_jyc_2026_instruments':pd.DataFrame([(d+pd.Timedelta(minutes=m),s) for d in tiny_days for m in minutes for s in tiny_stocks],columns=['date','instrument'])}
for frequency in data.FREQUENCIES:
    times=pd.to_datetime(data.regular_times(tiny_days,frequency))
    frame=pd.DataFrame([(t,s) for t in times for s in tiny_stocks],columns=['date','instrument'])
    for field in modules['v1002_preprocessing'].FIELDS:
        frame[field]=rng.uniform(1,20,len(frame))
    tables['fixture_'+frequency]=frame
sources={f:'fixture_'+f for f in data.FREQUENCIES}
shared=FakeDai(tables)
guard.query=shared.query
captured=[]
original_combine=modules['v1401_fast'].combine_scores
def capture_components(keys,scores):
    captured.append((keys.copy(),np.asarray(scores).copy()))
    return original_combine(keys,scores)
modules['v1401_fast'].combine_scores=capture_components
result=scope['main'](sources,'2024-01-04','2024-01-04')
modules['v1401_fast'].combine_scores=original_combine
parents=[]
counts=[]
for bundle in bundles:
    separate=FakeDai(tables)
    version=bundle['config']['solution_id']
    parents.append(modules[version+'_predict'].predict(sources,'2024-01-04','2024-01-04',bundle=bundle,dai_module=separate,device='cpu'))
    counts.append(len(separate.queries))
reference_public=scope['combine_predictions'](*parents).sort_values(['date','instrument']).reset_index(drop=True)
pd.testing.assert_frame_equal(result[['date','instrument']],reference_public[['date','instrument']])
assert len(captured)==1
raw_component_differences=[]
for index,parent in enumerate(parents):
    compared=captured[0][0].assign(actual=captured[0][1][:,index]).merge(parent,on=['date','instrument'],validate='one_to_one')
    np.testing.assert_allclose(compared.actual,compared.factor,atol=2e-7,rtol=1e-4)
    raw_component_differences.append(float(np.max(np.abs(compared.actual-compared.factor))))
# 与窗口对照使用同一容差；小截面标准化会放大已核验的 FP32 原始分数误差。
np.testing.assert_allclose(result.factor,reference_public.factor,atol=2e-5,rtol=1e-4)
assert list(result.columns)==['date','instrument','factor'] and len(result)==24
assert result.factor.dtype==np.dtype('float64') and np.isfinite(result.factor).all()
assert not result.duplicated(['date','instrument']).any()
assert pd.api.types.is_datetime64_any_dtype(result.date)
assert result.instrument.map(lambda x:isinstance(x,str)).all()
assert len(shared.queries)<sum(counts)
assert not any('vwap' in query or 'exposure' in query for query in shared.queries)
try:
    scope['reproduce']('must-not-be-created','unused')
except ValueError:
    pass
else:
    raise AssertionError('training guard missing')
assert not (package/'must-not-be-created').exists()
print(json.dumps({'synthetic_window_rows':len(keys),'same_backend_max_abs_difference':float(np.max(np.abs(actual-reference))),
    'all_eight_clocks_present':len(keys.date.unique())==8,'missing_keys_preserved':True,
    'public_main_synthetic_rows':len(result),'public_main_max_abs_difference':float(np.max(np.abs(result.factor-reference_public.factor))),
    'public_main_raw_component_max_abs_differences':raw_component_differences,
    'shared_queries':len(shared.queries),'separate_component_queries':counts,'labels_queried':False,
    'notebook_import_without_query':True,'training_guard_verified':True,'isolated_three_file_inference':True,
    'repository_code_on_import_path':False,'real_market_query_executed':False}))
'''


def verify(run, package):
    manifest_path = ROOT / 'experiments' / (package.name + '-submission.json')
    manifest = read_json(manifest_path)
    expected = {'v1401-submission.ipynb','v1002-model.json','v1101-model.json'}
    if set(path.name for path in package.iterdir()) != expected:
        raise ValueError('提交目录必须恰含三个规定附件')
    for path in package.iterdir():
        path.read_bytes().decode('utf-8')
        if path.stat().st_size > 50_000_000 or sha256(path) != manifest['files'][path.name]:
            raise ValueError('附件大小或字节指纹不符')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if sha256(run / 'best.pt') != manifest['model_sha256']:
        raise ValueError('组合权重已改变')
    bundles = [read_json(package / name) for name in ('v1002-model.json','v1101-model.json')]
    if bundles[0]['normalization'] != bundles[1]['normalization']:
        raise ValueError('统计量不同')
    restored_names = set()
    for index, (bundle, decode) in enumerate(zip(bundles,(decode_temporal,decode_structured))):
        if bundle['config'] != checkpoint['component_configs'][index]:
            raise ValueError('组件配置与冻结权重不同')
        for name, tensor in decode(bundle).items():
            key = f'components.{index}.' + name
            original = checkpoint['state_dict'][key]
            if tensor.dtype != original.dtype or tensor.shape != original.shape or not torch.equal(tensor, original):
                raise ValueError('实际参数附件恢复不一致: ' + key)
            restored_names.add(key)
    if restored_names != set(checkpoint['state_dict']):
        raise ValueError('组合权重有未恢复张量')
    book = read_json(package / 'v1401-submission.ipynb')
    scope = {'__name__':'submission_import','__file__':str(package/'submission.py')}
    for cell in book['cells']:
        if cell['cell_type']=='code':
            if cell['outputs'] or cell['execution_count'] is not None:
                raise ValueError('Notebook 保留了执行输出')
            code = ''.join(cell['source'])
            ast.parse(code)
            exec(compile(code,'<notebook>','exec'),scope)
    if scope['_CONFIG'] != checkpoint['config']:
        raise ValueError('Notebook 组合配置与冻结权重不同')
    for name, entry in scope['_OWN_SOURCES'].items():
        if hashlib.sha256(entry['text'].encode('utf-8')).hexdigest() != manifest['embedded_source_hashes'][name]:
            raise ValueError('组合源码指纹不符')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               SUBMISSION_REPOSITORY_ROOT=str(ROOT), PYTHONWARNINGS='ignore::DeprecationWarning')
    env.pop('PYTHONPATH', None)
    flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with tempfile.TemporaryDirectory(prefix='v1401-submit-') as temporary:
        isolated = Path(temporary) / 'package'
        isolated.mkdir()
        for name in expected:
            shutil.copyfile(package/name,isolated/name)
        result = subprocess.run([sys.executable,'-I','-c',ISOLATED_CHECK],cwd=isolated,env=env,capture_output=True,
                                text=True,encoding='utf-8',timeout=120,creationflags=flags)
        if result.returncode:
            raise RuntimeError(result.stderr)
        checked = json.loads(result.stdout.strip().splitlines()[-1])
        for bundle in bundles:
            version = bundle['config']['solution_id']
            project = Path(temporary)/version
            for relative in bundle['training_sources']:
                path = project/relative
                if project.resolve() not in path.resolve().parents:
                    raise ValueError('非法复现路径')
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(bundle['sources'][relative]['text'].encode('utf-8'))
            code = ("import sys,json,pathlib,site; sys.path.append(site.getusersitepackages()); p=pathlib.Path.cwd(); "
                    f"sys.path.insert(0,str(p/'solutions/{version}')); import {version}_train as t; "
                    "load=lambda x:json.loads(x.read_text(encoding='utf-8')); "
                    f"t.validate_training(load(p/'solutions/{version}/model-config.json'),load(p/'evaluation/protocols/local-v2-ic.json')); print('ok')")
            result = subprocess.run([sys.executable,'-I','-c',code],cwd=project,env=env,capture_output=True,
                                    text=True,encoding='utf-8',timeout=60,creationflags=flags)
            if result.returncode or 'ok' not in result.stdout:
                raise RuntimeError(result.stderr)
    report = {'checked_at':now(),'passed':True,'device':'cpu','weights_bit_exact':True,
        'scope':'实际双组件文本权重精确恢复、隔离合成窗口及完整合成 main 对照、独立组件训练源码导入；无长行情重推或性能基准。',
        'synthetic_check':checked,'component_training_imports_passed':['v1002','v1101'],
        'formal_training_performed':False,'market_interval_parity_repeated':False,'speed_benchmark_performed':False,
        'platform_execution_verified':False,'attachment_bytes':manifest['attachment_bytes'],
        'logical_review':['两组件输入、归一化与因果前缀相同，预处理只执行一次。',
            '两个网络保留独立冻结权重，各自共享股票日编码；全部请求股票计算后按完整时点组合。',
            '平台仅读取三个提交附件及授权行情，不读取标签、仓库、离线预测或外部权重。']}
    manifest.update(validation_status='local_passed_simplified',verification_script_sha256=sha256(__file__),validation=report,validations=[report])
    write_json(manifest_path,manifest)
    write_json(run/'text-submission-manifest.json',manifest)
    write_json(run/'submission-validation.json',report)
    return report


def record(run, package, report):
    path = ROOT/'experiments/submissions.csv'
    index = pd.read_csv(path,dtype=str,keep_default_na=False)
    manifest_path = ROOT/'experiments'/(package.name+'-submission.json')
    row = {'submission_id':package.name,'run_id':run.name,'status':'local_verified_not_submitted',
           'directory':package.relative_to(ROOT).as_posix(),'manifest_sha256':sha256(manifest_path),
           'platform_submission_id':'','platform_score':'','manifest_path':manifest_path.relative_to(ROOT).as_posix()}
    pd.concat([index.loc[index.submission_id!=package.name],pd.DataFrame([row])],ignore_index=True).to_csv(path,index=False,encoding='utf-8-sig')
    summary_path = ROOT/'experiments/v1401-gru.json'
    summary = read_json(summary_path)
    summary.update(exported=True,submission_directory=row['directory'],submission_validation=report,submission_validations=[report],
        submission_manifest_sha256=row['manifest_sha256'],submission_manifest_path=row['manifest_path'],export_purpose='local_improvement')
    write_json(summary_path,summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--record',action='store_true')
    args=parser.parse_args()
    report=verify(args.run_dir.resolve(),args.package.resolve())
    if args.record:
        record(args.run_dir.resolve(),args.package.resolve(),report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
