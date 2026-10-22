"""按简化约定检查 v1101 文本材料；只做隔离合成推理，不重复长行情评估。"""

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
sys.path.insert(0, str(ROOT / 'common'))
sys.path.insert(0, str(ROOT / 'solutions/v1101'))
from artifacts import now, read_json, sha256, write_json
from v1101_text_artifact import decode_state, state_digest


ISOLATED_CHECK = r'''
import json, pathlib, sys, types, os, site
sys.path.append(site.getusersitepackages())
import torch
import numpy as np
import pandas as pd
torch.set_num_threads(4)
package = pathlib.Path.cwd()
assert set(p.name for p in package.iterdir()) == {'v1101-submission.ipynb', 'v1101-model.json'}
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
book = json.loads((package/'v1101-submission.ipynb').read_text(encoding='utf-8'))
scope = {'__name__': 'submission_import', '__file__': str(package/'submission.py')}
for cell in book['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), '<notebook>', 'exec'), scope)
assert scope['_RUNTIME'] is None
bundle, modules = scope['_runtime']()
data = modules['v1101_data']
model = modules['v1101_model'].MultiFrequencyGRU(bundle['config']).eval()
model.load_state_dict(modules['v1101_text_artifact'].decode_state(bundle))
rng = np.random.default_rng(901)
days = pd.bdate_range('2024-01-02', periods=22)
stocks = pd.Index(['S0', 'S1', 'S2'])
grids = {}
for frequency in data.FREQUENCIES:
    times = data.regular_times(days, frequency)
    values = rng.normal(size=(len(times), len(stocks), 72)).astype(np.float32)
    values[3, 1] = np.nan
    grids[frequency] = times, values
minutes = (570, 600, 630, 660, 690, 810, 840, 870)
keys = pd.DataFrame([(days[-1]+pd.Timedelta(minutes=m), s) for m in minutes for s in stocks], columns=['date', 'instrument'])
keys = keys.drop(index=[2, 4, 11]).sample(frac=1, random_state=4).reset_index(drop=True)
block = data.WindowBlock(keys, grids, stocks, bundle['config'], days.to_numpy())
actual = modules['v1101_fast'].predict_day(model, block, torch.device('cpu'), stock_batch_size=2)
windows, clocks = block.batch(np.arange(len(keys)))
with torch.inference_mode():
    reference = model({f:torch.from_numpy(x) for f,x in windows.items()}, torch.from_numpy(clocks)).numpy()
np.testing.assert_allclose(actual, reference, atol=2e-7, rtol=1e-4)
ten = np.flatnonzero(keys.date.dt.strftime('%H:%M').eq('10:00'))[0]
assert np.any(windows['bar5m'][ten, :72, 240:246] != 0)
assert np.any(windows['bar30m'][ten, :72, 160:161] != 0)
frame = keys.assign(factor=actual)
assert list(frame.columns) == ['date','instrument','factor']
assert frame.factor.dtype == np.dtype('float64') and np.isfinite(frame.factor).all()
assert pd.api.types.is_datetime64_any_dtype(frame.date)
assert frame.instrument.map(lambda x:isinstance(x,str)).all()
assert not frame.duplicated(['date','instrument']).any()
assert len(frame) == len(keys)
print(json.dumps({'synthetic_rows':len(frame), 'same_backend_max_abs_difference':float(np.max(np.abs(actual-reference))),
    'inference_device':'cpu', 'all_clocks_present':len(frame.date.unique())==8,
    'missing_requested_keys_preserved':True, 'current_day_coarse_inputs_present':True,
    'notebook_import_without_query':True, 'isolated_two_file_model_inference':True,
    'repository_code_on_import_path':False, 'public_main_market_query_executed':False}))
'''


def manifest_path(package):
    return ROOT / 'experiments' / (package.name + '-submission.json')


def verify(run, package):
    run, package = Path(run).resolve(), Path(package).resolve()
    manifest = read_json(manifest_path(package))
    expected = {'v1101-submission.ipynb', 'v1101-model.json'}
    if set(p.name for p in package.iterdir()) != expected:
        raise ValueError('材料必须仅包含指定 Notebook 和 JSON')
    for path in package.iterdir():
        path.read_bytes().decode('utf-8')
        if path.stat().st_size > 50_000_000 or sha256(path) != manifest['files'][path.name]:
            raise ValueError('附件大小或指纹不符')
    bundle = read_json(package / 'v1101-model.json')
    book = read_json(package / 'v1101-submission.ipynb')
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            if cell['outputs'] or cell['execution_count'] is not None:
                raise ValueError('Notebook 不得保留执行输出')
            ast.parse(''.join(cell['source']))
    if 'if __name__ == "__main__":' not in ''.join(book['cells'][-1]['source']):
        raise ValueError('缺少受保护的调试单元格')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    restored = decode_state(bundle)
    if sha256(run / 'best.pt') != manifest['model_sha256']:
        raise ValueError('冻结权重已改变')
    if checkpoint['config'] != bundle['config']:
        raise ValueError('配置与冻结权重不一致')
    for name, original in checkpoint['state_dict'].items():
        if name not in restored or original.dtype != restored[name].dtype or not torch.equal(original, restored[name]):
            raise ValueError('实际附件权重恢复不一致: ' + name)
    if state_digest(restored) != bundle['tensor_sha256']:
        raise ValueError('实际附件张量位模式不一致')
    for name, source in bundle['sources'].items():
        if hashlib.sha256(source['text'].encode('utf-8')).hexdigest() != manifest['embedded_source_hashes'][name]:
            raise ValueError('内嵌源码不一致: ' + name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               SUBMISSION_REPOSITORY_ROOT=str(ROOT))
    env.pop('PYTHONPATH', None)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with tempfile.TemporaryDirectory(prefix='v1101-submit-') as directory:
        isolated = Path(directory) / 'package'
        isolated.mkdir()
        for name in expected:
            shutil.copyfile(package / name, isolated / name)
        result = subprocess.run([sys.executable, '-I', '-c', ISOLATED_CHECK], cwd=isolated, env=env,
                                check=False, capture_output=True, text=True, encoding='utf-8', timeout=120, creationflags=flags)
        if result.returncode:
            raise RuntimeError(result.stderr)
        checked = json.loads(result.stdout.strip().splitlines()[-1])
        project = Path(directory) / 'reproduction'
        for relative in bundle['training_sources']:
            path = project / relative
            if project.resolve() not in path.resolve().parents:
                raise ValueError('非法复现源码路径')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle['sources'][relative]['text'].encode('utf-8'))
        code = ("import sys,json,pathlib,site; sys.path.append(site.getusersitepackages()); p=pathlib.Path.cwd(); "
                "sys.path.insert(0,str(p/'solutions/v1101')); import v1101_train as t; "
                "load=lambda x:json.loads(x.read_text(encoding='utf-8')); "
                "t.validate_training(load(p/'solutions/v1101/model-config.json'),load(p/'evaluation/protocols/local-v2-ic.json')); "
                "print('standalone_training_import_passed')")
        smoke = subprocess.run([sys.executable, '-I', '-c', code], cwd=project, env=env, check=False,
                               capture_output=True, text=True, encoding='utf-8', timeout=60, creationflags=flags)
        if smoke.returncode:
            raise RuntimeError(smoke.stderr)
        if 'standalone_training_import_passed' not in smoke.stdout:
            raise ValueError('独立训练源码导入检查未完成')
    report = {
        'checked_at': now(), 'passed': True, 'device': 'cpu',
        'scope': '简化转换核验：逻辑审查、实际文本权重精确恢复、隔离合成窗口 fast／逐样本对照及独立复现源码导入；未重跑长行情区间或性能基准。',
        'weights_bit_exact': True, 'normalization_precision': 'unchanged',
        'model_sha256': manifest['model_sha256'], 'tensor_sha256': bundle['tensor_sha256'],
        'synthetic_check': checked, 'standalone_training_sources_import_passed': True,
        'formal_training_performed': False, 'market_interval_parity_repeated': False, 'speed_benchmark_performed': False,
        'logical_review': ['归一化和股票池处理沿用既有 fast 路径；只用冻结源码。',
            'Calendar 和 WindowBlock 按每频 include_today 取窗，超过预测时点的 Bar 全部置零。',
            '字段结构编码只在同一 Bar 内交互；三频历史与当天前缀共享，跨频交互在同一时点内，最终仅取收益主头。',
            '完整股票日只编码一次历史和当前序列；具体速度倍数未测量。'],
        'prior_evidence': ['experiments/readiness/v1101.json', 'tests/test_v1101.py', 'tests/test_v1101_entrypoint.py',
                           'docs/v303-inference-fast.md'],
        'attachment_bytes': manifest['attachment_bytes'],
        'platform_execution_verified': False,
    }
    manifest.update(validation_status='local_passed_simplified', verification_script_sha256=sha256(__file__),
                    validation=report, validations=[report])
    write_json(manifest_path(package), manifest)
    write_json(run / 'text-submission-manifest.json', manifest)
    write_json(run / 'submission-validation.json', report)
    return report


def record(run, package, report):
    path = ROOT / 'experiments/submissions.csv'
    index = pd.read_csv(path, dtype=str, keep_default_na=False)
    row = {'submission_id':package.name, 'run_id':run.name, 'status':'local_verified_not_submitted',
           'directory':package.relative_to(ROOT).as_posix(), 'manifest_sha256':sha256(manifest_path(package)),
           'platform_submission_id':'', 'platform_score':'', 'manifest_path':manifest_path(package).relative_to(ROOT).as_posix()}
    index = pd.concat([index.loc[index.submission_id != package.name], pd.DataFrame([row])], ignore_index=True)
    index.to_csv(path, index=False, encoding='utf-8-sig')
    summary_path = ROOT / 'experiments/v1101-gru.json'
    summary = read_json(summary_path)
    manifest = read_json(manifest_path(package))
    summary.update(exported=True, submission_directory=row['directory'], submission_validation=report,
                   submission_validations=[report], submission_manifest_sha256=row['manifest_sha256'],
                   submission_manifest_path=row['manifest_path'], export_purpose=manifest['export_purpose'],
                   export_reason=manifest['export_reason'])
    write_json(summary_path, summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = verify(args.run_dir, args.package)
    if args.record:
        record(args.run_dir.resolve(), args.package.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
