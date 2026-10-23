"""显式复现：原始授权数据预处理、父网络从头训练、再训练树补充。"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def reproduce(bundle, workspace, data_root, *, training_confirmed=False):
    if not training_confirmed:raise ValueError('训练复现尚未明确确认')
    workspace,data=Path(workspace).resolve(),Path(data_root).resolve()
    if not data.is_dir():raise FileNotFoundError(data)
    project=workspace/'source'
    sources={}
    for relative in bundle['training_sources']:
        path=(project/relative).resolve()
        if not path.is_relative_to(project) or path==project:raise ValueError('非法复现路径')
        sources[path]=bundle['sources'][relative]['text'].encode('utf-8')
    workspace.mkdir(parents=True,exist_ok=False)
    for path,value in sources.items():
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    solution=project/'solutions/v2801'
    training,holdout,parent,run=(workspace/name for name in ('preprocessing-train','preprocessing-holdout','parent-run','run'))
    protocol=project/'evaluation/protocols/local-v2-ic.json'
    commands=[
        ['v2401_prepare.py','--run-dir',str(training),'--data-root',str(data),'--protocol',str(protocol)],
        ['v2401_apply.py','--training-run',str(training),'--run-dir',str(holdout),'--data-root',str(data)],
        ['v2401_train.py','--config',str(solution/'parent-model-config.json'),'--protocol',str(protocol),
         '--training-run',str(training),'--holdout-run',str(holdout),'--data-root',str(data),
         '--run-dir',str(parent),'--training-confirmed','--reproduction-only'],
    ]
    for name,*arguments in commands:
        subprocess.run([sys.executable,str(solution/name),*arguments],cwd=project,check=True)
    config=copy.deepcopy(bundle['config'])
    config['parent_run_id']=parent.name
    config['model']['parent_model_sha256']=hashlib.sha256((parent/'best.pt').read_bytes()).hexdigest()
    config['backbone_config']=json.loads((parent/'config.json').read_text(encoding='utf-8'))
    config_path=workspace/'tree-config.json'
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,str(solution/'v2801_train.py'),'--config',str(config_path),
        '--protocol',str(protocol),'--parent-run',str(parent),'--training-run',str(training),
        '--holdout-run',str(holdout),'--data-root',str(data),'--run-dir',str(run),
        '--training-confirmed','--reproduction-only'],cwd=project,check=True)
    return run
