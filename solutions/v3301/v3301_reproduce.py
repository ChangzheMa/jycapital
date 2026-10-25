"""显式从头复现父模型及独立时点训练；正常平台推理不调用。"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def reproduce(bundle, workspace, data_root, *, training_confirmed=False):
    if not training_confirmed:
        raise ValueError('训练复现须显式确认')
    workspace, data = Path(workspace).resolve(), Path(data_root).resolve()
    if not data.is_dir():
        raise FileNotFoundError(data)
    project = workspace / 'source'
    sources = {}
    for relative in bundle['training_sources']:
        path = (project / relative).resolve()
        if not path.is_relative_to(project) or path == project:
            raise ValueError('非法复现路径')
        sources[path] = bundle['sources'][relative]['text'].encode('utf-8')
    workspace.mkdir(parents=True, exist_ok=False)
    for path, value in sources.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    solution = project / 'solutions/v3301'
    parent_solution = project / 'solutions/v3201'
    parent_solution.mkdir()
    # 父训练入口内部按v3201冻结源码，因此为它恢复正确的版本目录。
    for p in solution.iterdir():
        if p.name.startswith(('v3201_', 'v1601_')) or p.name in ('config.json', 'requirements.txt', 'requirements-training.txt', 'requirements-inference.txt'):
            (parent_solution / p.name).write_bytes(p.read_bytes())
    parent_config = bundle['config']['backbone_config']
    (parent_solution / 'model-config.json').write_text(json.dumps(parent_config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    training, holdout, parent, run = (workspace / n for n in ('preprocessing-train', 'preprocessing-holdout', 'parent-run', 'run'))
    protocol = project / 'evaluation/protocols/local-v2-ic.json'
    commands = [
        [parent_solution / 'v3201_prepare.py', '--run-dir', training, '--data-root', data, '--protocol', protocol],
        [parent_solution / 'v3201_apply.py', '--training-run', training, '--run-dir', holdout, '--data-root', data],
        [parent_solution / 'v3201_train.py', '--config', parent_solution / 'model-config.json', '--protocol', protocol,
         '--training-run', training, '--holdout-run', holdout, '--data-root', data, '--run-dir', parent,
         '--training-confirmed', '--reproduction-only'],
    ]
    for command in commands:
        subprocess.run([sys.executable, *map(str, command)], cwd=project, check=True)
    config = copy.deepcopy(bundle['config'])
    config['parent_run_id'] = parent.name
    config['model']['parent_model_sha256'] = hashlib.sha256((parent / 'best.pt').read_bytes()).hexdigest()
    config_path = workspace / 'clock-config.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, str(solution / 'v3301_train.py'), '--config', str(config_path),
        '--protocol', str(protocol), '--parent-run', str(parent), '--training-run', str(training),
        '--holdout-run', str(holdout), '--data-root', str(data), '--run-dir', str(run),
        '--training-confirmed', '--reproduction-only'], cwd=project, check=True)
    return run
