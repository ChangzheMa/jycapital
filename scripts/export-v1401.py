"""导出冻结双模型的共享预处理 fast 材料；两个参数附件逐字节复用。"""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))
from artifacts import read_json, write_json, sha256, now


def export(run):
    manifest = read_json(run / 'manifest.json')
    config = read_json(run / 'config.json')
    comparison = read_json(run / 'comparison.json')
    if manifest['status'] != 'complete' or manifest['solution_id'] != 'v1401' or not comparison['candidate']:
        raise ValueError('只导出已完成且通过本地晋级条件的 v1401')
    if sha256(run / 'best.pt') != manifest['selection']['model_sha256']:
        raise ValueError('组合冻结权重已改变')
    if comparison['config_sha256'] != sha256(run / 'config.json'):
        raise ValueError('组合配置与评估不一致')
    package = ROOT / 'submissions' / (run.name + '-fast')
    if package.exists():
        raise FileExistsError(package)
    assets, bundles = [], []
    for item in config['model']['components']:
        parent_package = ROOT / 'submissions' / (item['run_id'] + '-fast')
        parent_manifest = read_json(ROOT / 'experiments' / (parent_package.name + '-submission.json'))
        name = item['solution_id'] + '-model.json'
        path = parent_package / name
        if parent_manifest['model_sha256'] != item['model_sha256'] or sha256(path) != parent_manifest['files'][name]:
            raise ValueError('组件文本材料与冻结权重不对应')
        if path.stat().st_size > 50_000_000:
            raise ValueError('组件参数附件超过50 MB')
        assets.append(path)
        bundles.append(read_json(path))
    if bundles[0]['normalization'] != bundles[1]['normalization'] or bundles[0]['config']['data'] != bundles[1]['config']['data']:
        raise ValueError('组件不能共享输入和统计量')
    for key in ('data', 'processing'):
        if bundles[0]['preprocessing'][key] != bundles[1]['preprocessing'][key]:
            raise ValueError('组件预处理不一致')
    sources, overrides = {}, {}
    for name in ('v1401_combine.py', 'v1401_model.py', 'v1401_fast.py', 'v1401_predict.py', 'v1401_notebook.py'):
        relative = 'solutions/v1401/' + name
        if relative in manifest['source']['files']:
            path = run / 'source' / relative
            if sha256(path) != manifest['source']['files'][relative]:
                raise ValueError('冻结组合源码已改变')
        else:
            path = run / 'runtime-source' / relative
            current = ROOT / relative
            if path.exists() and path.read_bytes() != current.read_bytes():
                raise ValueError('已冻结运行适配器发生变化，须显式处理')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                shutil.copyfile(current, path)
            overrides[relative] = {'source_path': path.relative_to(ROOT).as_posix(), 'sha256': sha256(path)}
        sources[relative] = {'text': path.read_bytes().decode('utf-8'), 'sha256': sha256(path)}
    template = sources.pop('solutions/v1401/v1401_notebook.py')['text']
    entry = template.replace('_CONFIG = None  # EXPORT_CONFIG', '_CONFIG = ' + repr(config))
    entry = entry.replace('_OWN_SOURCES = None  # EXPORT_SOURCES', '_OWN_SOURCES = ' + repr(sources))
    description = '''# v1401 冻结双模型 fast 提交

上传本 Notebook，以及同目录 `v1002-model.json`、`v1101-model.json`，共三个 UTF-8 文本文件。
两个 JSON 均为原组件的完整精度权重、配置、统计量和训练／推理源码，不含离线预测答案。

组件为 v1002 第 4 轮 EMA 与 v1101 第 5 轮 EMA；组合规则在本次评估前固定，不新增训练、不搜索权重。
每个预测时点，分别把两套完整股票池预测按总体标准差标准化，再等权平均；常数组件映射为零。
factor 是排序分数，不是校准收益。全部股票计算后才组合，不在小批量内计算截面标准化。

`main(datasources, start_date, end_date)` 使用 `bar1m`、`bar5m`、`bar30m` 的逻辑行情源。
共享一次行情读取、归一化和切窗；两个冻结网络分别按股票日共享历史及当天因果前缀。
网络总参数 4,715,286，网络推理成本高于单个组件；没有测量平台运行时间，不承诺平台速度倍数。
结束日期字符串包含全天，结束时间戳包含该分钟。导入不查询行情，推理不训练、不读取标签、不访问外网。

依赖及完整训练代码在各 JSON 的 requirements、sources、training_sources 中；推理还需要平台 dai。
显式调用 `reproduce("reproduced-v1401", "/path/to/data", training_confirmed=True)` 会串行复现两个组件，
返回各自运行目录；目标目录必须不存在。重新训练受硬件与数值环境影响，不保证恢复同一权重。
复现后的两套完整预测可交给 `combine_predictions(left, right)`，使用同一固定组合函数。
通常仅调用 main 使用本次冻结权重即可。下方调试单元格不启动训练。
'''
    debug = '''if __name__ == "__main__":
    result = main({f: "cpt_jyc_2026_stock_" + f for f in ("bar1m", "bar5m", "bar30m")}, "2024-12-30", "2024-12-31")
    import numpy as np
    import pandas as pd
    assert list(result.columns) == ["date", "instrument", "factor"]
    assert not result.empty and result.factor.dtype == np.dtype("float64")
    assert pd.api.types.is_datetime64_any_dtype(result.date)
    assert result.instrument.map(lambda value: isinstance(value, str)).all()
    assert not result[["date", "instrument"]].isna().any().any()
    assert not result.duplicated(["date", "instrument"]).any()
    assert np.isfinite(result.factor).all()
    print("输出行数：", len(result))
    print(result.dtypes)
    display(result.head())
'''
    cells = [{'cell_type': 'markdown', 'id': 'v1401-usage', 'metadata': {}, 'source': description.splitlines(keepends=True)}]
    cells += [{'cell_type': 'code', 'id': 'v1401-' + name, 'metadata': {}, 'execution_count': None, 'outputs': [],
               'source': code.splitlines(keepends=True)} for name, code in (('inference', entry), ('debug', debug))]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python', 'version': '3.11'}}, 'nbformat': 4, 'nbformat_minor': 5}
    package.mkdir(parents=True)
    for asset in assets:
        shutil.copyfile(asset, package / asset.name)
    write_json(package / 'v1401-submission.ipynb', notebook)
    report = {'solution_id': 'v1401', 'run_id': run.name, 'exported_at': now(),
        'layout': 'one-notebook-two-component-json', 'max_attachment_bytes': 50_000_000,
        'model_sha256': manifest['selection']['model_sha256'], 'components': config['model']['components'],
        'component_parameter_files_reused_byte_exact': True, 'normalization_precision': 'unchanged',
        'source_snapshot_hashes': manifest['source']['files'], 'runtime_overrides': overrides,
        'embedded_source_hashes': {name: entry['sha256'] for name, entry in sources.items()},
        'notebook_template_sha256': sha256(run / 'runtime-source/solutions/v1401/v1401_notebook.py'),
        'export_script_sha256': sha256(__file__), 'comparison_sha256': sha256(run / 'comparison.json'),
        'local_candidate': True, 'local_decision': comparison['decision'], 'export_purpose': 'local_improvement',
        'export_reason': '预声明的冻结等权组合提高验证 IC，双集覆盖完整；增加双网络推理成本。',
        'additional_training': False, 'inference_execution': {'method': 'shared_preprocessing_two_frozen_fast_backbones',
            'stock_batch_size': config['inference']['stock_day_batch_size'], 'aggregation_scope': 'complete_timestamp_cross_section'},
        'platform_status': 'not_submitted', 'validation_status': 'pending',
        'files': {path.name: sha256(path) for path in sorted(package.iterdir())},
        'attachment_bytes': {path.name: path.stat().st_size for path in sorted(package.iterdir())}}
    if any(size > 50_000_000 for size in report['attachment_bytes'].values()):
        raise ValueError('附件超过单文件限制')
    write_json(ROOT / 'experiments' / (package.name + '-submission.json'), report)
    write_json(run / 'text-submission-manifest.json', report)
    return package, report['attachment_bytes']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(export(args.run_dir.resolve()))
