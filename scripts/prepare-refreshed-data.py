"""从已完整核对的新原始快照建立独立预处理；不启动或恢复模型训练。"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
from artifacts import now,read_json,sha256
from research_queue import atomic_json,source_files


def validate_comparison(comparison,new_root):
    report=read_json(comparison)
    if report.get('status')!='complete' or report.get('covers_required_protocol') is not True:
        raise ValueError('必须先完成覆盖固定协议和所需表的比较')
    if Path(report['identity']['new_root']).resolve()!=new_root.resolve():
        raise ValueError('新数据目录与比较来源不同')
    manifest=new_root/'download-manifest.json'
    if sha256(manifest)!=report['identity']['new_manifest_sha256']:
        raise ValueError('下载清单在比较后发生变化')
    source=read_json(manifest)
    if source['status']!='complete':raise ValueError('新下载尚未完整')
    if sha256(ROOT/'evaluation/protocols/local-v2-ic.json')!=report['identity']['protocol_sha256']:
        raise ValueError('固定评估协议发生变化')
    if sha256(ROOT/'scripts/compare-raw-snapshots.py')!=report['identity']['script_sha256']:
        raise ValueError('比较实现已变化，须先核对对应的冻结比较证据')
    # 比较过程已逐文件核验完整SHA；此处再次核对大小，预处理仍独立记录所读文件的SHA。
    for relative,item in source['files'].items():
        path=(new_root/relative).resolve()
        if not path.is_relative_to(new_root.resolve()) or not path.is_file() or path.stat().st_size!=item['bytes']:
            raise ValueError('新数据文件缺失或大小发生变化：'+relative)
    return report


def processing_config(original,threads,memory_limit,batch_rows):
    import copy
    result=copy.deepcopy(original)
    if type(threads) is not int or threads<1 or type(batch_rows) is not int or batch_rows<1:
        raise ValueError('处理线程和批量必须为正整数')
    result['processing'].update(threads=threads,memory_limit=memory_limit,batch_rows=batch_rows)
    return result


def stage_sources(workspace,solution):
    project=workspace/'preprocessing-source'
    fingerprints=source_files(ROOT,solution)
    for relative,digest in fingerprints.items():
        source=ROOT/relative;destination=project/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if sha256(destination)!=digest:raise ValueError('预处理执行快照已存在且内容不同：'+relative)
        else:shutil.copyfile(source,destination)
        if sha256(destination)!=digest:raise ValueError('冻结复制过程中源文件改变：'+relative)
    return project,fingerprints


def commands(python,project,solution,new_root,preprocessing_root,config_path):
    protocol=project/'evaluation/protocols/local-v2-ic.json'
    train,holdout=preprocessing_root/'train',preprocessing_root/'holdout'
    return [('train',train,[str(python),'-u',str(project/f'solutions/{solution}/{solution}_prepare.py'),
        '--config',str(config_path),'--protocol',str(protocol),'--data-root',str(new_root),'--run-dir',str(train)]),
        ('holdout',holdout,[str(python),'-u',str(project/f'solutions/{solution}/{solution}_apply.py'),
         '--training-run',str(train),'--run-dir',str(holdout),'--data-root',str(new_root)])]


def verify_complete(folder,new_root):
    manifest=read_json(folder/'manifest.json')
    if manifest['status']!='complete':raise ValueError('预处理未完成：'+str(folder))
    if sha256(folder/'summary.json')!=manifest['summary_sha256']:
        raise ValueError('预处理摘要指纹不符')
    if sha256(folder/'data-manifest.json')!=manifest['data_manifest_sha256']:
        raise ValueError('预处理输入指纹不符')
    data=read_json(folder/'data-manifest.json')
    if Path(data['root']).resolve()!=new_root.resolve():raise ValueError('预处理没有绑定新原始目录')
    expected=read_json(new_root/'download-manifest.json')['files']
    for relative,item in data['files'].items():
        if relative not in expected or item['sha256']!=expected[relative]['sha256'] or item['bytes']!=expected[relative]['bytes']:
            raise ValueError('预处理实际读取内容不属于已核对的新快照：'+relative)
    universe=data['universe']
    if universe['sha256']!=expected[universe['path']]['sha256']:
        raise ValueError('预处理股票池不属于新快照')
    return {'path':str(folder),'manifest_sha256':sha256(folder/'manifest.json'),
            'data_manifest_sha256':sha256(folder/'data-manifest.json'),
            'summary_sha256':sha256(folder/'summary.json')}


def prepare(new_root,comparison,workspace,preprocessing_root,*,solution='v3001',threads=2,
            memory_limit='1GB',batch_rows=32768,resume=False):
    new_root,comparison,workspace,preprocessing_root=[Path(p).resolve() for p in (new_root,comparison,workspace,preprocessing_root)]
    if preprocessing_root==new_root or preprocessing_root.is_relative_to(new_root):
        raise ValueError('预处理目录必须与原始快照分开')
    if workspace==new_root or workspace.is_relative_to(new_root):raise ValueError('执行记录不能写进原始快照')
    proof=validate_comparison(comparison,new_root)
    workspace.mkdir(parents=True,exist_ok=True)
    plan_path=workspace/'preprocessing-plan.json'
    config_path=workspace/'preprocessing-config.json'
    if plan_path.exists():
        if not resume:raise FileExistsError('预处理计划已存在；继续原计划须使用--resume')
        plan=read_json(plan_path)
        if (plan['raw_root']!=str(new_root) or plan['preprocessing_root']!=str(preprocessing_root)
                or plan['comparison_sha256']!=sha256(comparison) or plan['solution_id']!=solution):
            raise ValueError('续处理来源、输出目录或方案不同')
        if plan['driver_sha256']!=sha256(__file__):raise ValueError('预处理编排实现与原计划不同')
        project=Path(plan['project'])
        for relative,digest in plan['source_files'].items():
            if sha256(project/relative)!=digest:raise ValueError('冻结预处理源码改变：'+relative)
        if sha256(config_path)!=plan['processing_config_sha256']:raise ValueError('有效预处理配置改变')
    else:
        if preprocessing_root.exists() and any(preprocessing_root.iterdir()):
            raise FileExistsError('拒绝覆盖已有预处理目录')
        project,fingerprints=stage_sources(workspace,solution)
        original=read_json(project/f'solutions/{solution}/config.json')
        config=processing_config(original,threads,memory_limit,batch_rows)
        atomic_json(config_path,config)
        plan={'created_at':now(),'solution_id':solution,'project':str(project),'source_files':fingerprints,
            'driver_sha256':sha256(__file__),
            'raw_root':str(new_root),'raw_manifest_sha256':sha256(new_root/'download-manifest.json'),
            'comparison_path':str(comparison),'comparison_sha256':sha256(comparison),
            'preprocessing_root':str(preprocessing_root),'processing_config_sha256':sha256(config_path),
            'processing_limits':config['processing'],'existing_runs_modified':False,'formal_training_performed':False,
            'status':'prepared','completed_stages':{}}
        atomic_json(plan_path,plan)
    schedule=commands(Path(sys.executable),project,solution,new_root,preprocessing_root,config_path)
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    try:
        for stage,folder,command in schedule:
            if folder.exists():
                manifest=read_json(folder/'manifest.json')
                if manifest['status']=='complete':
                    plan['completed_stages'][stage]=verify_complete(folder,new_root)
                    atomic_json(plan_path,plan)
                    continue
                command.append('--resume')
            plan.update(status='running',stage=stage,last_started_at=now())
            atomic_json(plan_path,plan)
            subprocess.run(command,cwd=project,env=env,creationflags=flags,check=True)
            plan['completed_stages'][stage]=verify_complete(folder,new_root)
            atomic_json(plan_path,plan)
        train,holdout=preprocessing_root/'train',preprocessing_root/'holdout'
        config=read_json(config_path);protocol=read_json(project/'evaluation/protocols/local-v2-ic.json')
        for frequency in config['data']['frequencies']:
            relative='normalization/'+frequency+'.json'
            if sha256(train/relative)!=sha256(holdout/relative):raise ValueError('留出集统计量不是冻结训练统计')
            if read_json(train/relative)['fit_interval']!=protocol['splits']['train']:
                raise ValueError('归一化拟合区间不是固定训练区间')
        binding={'schema_version':1,'status':'prepared_for_scheduler','prepared_at':now(),
            'snapshot_id':read_json(new_root/'download-manifest.json')['snapshot_id'],
            'raw_root':str(new_root),'raw_manifest_sha256':plan['raw_manifest_sha256'],
            'comparison_path':str(comparison),'comparison_sha256':plan['comparison_sha256'],
            'protocol_sha256':proof['identity']['protocol_sha256'],
            'training_run':str(preprocessing_root/'train'),'holdout_run':str(preprocessing_root/'holdout'),
            'preprocessing':plan['completed_stages'],
            'cache_policy':'由新数据与新预处理指纹生成新键；不复用旧全局CACHE常量，不覆盖旧数组。',
            'existing_started_runs':'keep_original_snapshot_no_restart',
            'formal_training_performed':False,'controller_activation_verified':False}
        atomic_json(workspace/'data-binding.json',binding)
        plan.update(status='complete',stage='complete',completed_at=now());plan.pop('failure',None)
        atomic_json(plan_path,plan)
        return binding
    except BaseException as error:
        plan.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                    failure={'at':now(),'type':type(error).__name__})
        atomic_json(plan_path,plan)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-root',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--preprocessing-root',type=Path,required=True)
    parser.add_argument('--solution',default='v3001')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--memory-limit',default='1GB')
    parser.add_argument('--batch-rows',type=int,default=32768)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    result=prepare(args.new_root,args.comparison,args.workspace,args.preprocessing_root,
        solution=args.solution,threads=args.threads,memory_limit=args.memory_limit,batch_rows=args.batch_rows,resume=args.resume)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
