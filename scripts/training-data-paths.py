"""向串行调度器提供新运行的数据路径；已启动运行必须沿原命令，不适用此入口。"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'common'),str(ROOT/'scripts')]
from artifacts import read_json,sha256
from research_queue import safe_path,validate_spec,readiness_error


def resolve_new_run(root,solution_id):
    root=Path(root).resolve()
    directory=safe_path(root,'solutions/'+solution_id)
    candidate=validate_spec(root,directory/'experiment.json')
    run_id=candidate['run_id']
    if (root/'outputs/runs'/run_id).exists() or (root/'outputs/run-control'/run_id/'attempt.json').exists():
        raise ValueError('该运行已启动或已有启动记录；保留原数据和原恢复命令，不切换数据')
    if candidate['status']!='ready' or readiness_error(root,candidate) is not None:
        raise ValueError('只有指纹核验通过的ready新候选可以解析启动数据')
    policy=read_json(root/'experiments/research-policy.json')
    refresh=policy.get('data_refresh',{})
    binding_path=safe_path(root,refresh['future_data_binding'])
    binding=read_json(binding_path)
    if binding['status'] not in ('prepared_for_scheduler','active'):
        raise ValueError('新快照的训练和留出预处理尚未完成')
    raw=Path(binding['raw_root']).resolve()
    if raw!=safe_path(root,refresh['raw_snapshot_root']):raise ValueError('原始目录不属于用户指定的新快照')
    if sha256(raw/'download-manifest.json')!=binding['raw_manifest_sha256']:
        raise ValueError('新快照清单改变')
    raw_manifest=read_json(raw/'download-manifest.json')
    proof=Path(binding['comparison_path']).resolve()
    if sha256(proof)!=binding['comparison_sha256']:raise ValueError('新旧内容比较证据改变')
    compared=read_json(proof)
    if compared['status']!='complete' or not compared['covers_required_protocol']:
        raise ValueError('新旧内容比较未完整覆盖协议')
    if (compared['identity']['new_manifest_sha256']!=binding['raw_manifest_sha256'] or
            Path(compared['identity']['new_root']).resolve()!=raw or
            compared['identity']['protocol_sha256']!=binding['protocol_sha256']):
        raise ValueError('比较记录与所选新快照不一致')
    if sha256(root/'evaluation/protocols/local-v2-ic.json')!=binding['protocol_sha256']:
        raise ValueError('评估协议与准备时不同')
    protocol=read_json(root/'evaluation/protocols/local-v2-ic.json')
    evaluation_sources={table:raw_manifest['files'][table+'.parquet']['sha256']
                        for table in (protocol['universe_table'],protocol['label_table'])}
    registry=read_json(root/'experiments/data-snapshots.json')
    cohorts=[item for item in registry['snapshots'] if item['evaluation_sources']==evaluation_sources]
    if len(cohorts)!=1:raise ValueError('新训练评估数据必须对应唯一已登记快照')
    cohort=cohorts[0]
    if cohort.get('proof') and sha256(safe_path(root,cohort['proof']))!=cohort['proof_sha256']:
        raise ValueError('评估快照登记证据改变')
    import csv
    with (root/'experiments/runs.csv').open(encoding='utf-8-sig',newline='') as stream:
        comparable=[row['run_id'] for row in csv.DictReader(stream) if row.get('data_snapshot_id')==cohort['id']]
    for stage,key in (('train','training_run'),('holdout','holdout_run')):
        path=Path(binding[key]).resolve();record=binding['preprocessing'][stage]
        if path!=Path(record['path']).resolve():raise ValueError('预处理目录与绑定证据不一致')
        manifest=read_json(path/'manifest.json')
        if manifest['status']!='complete' or sha256(path/'manifest.json')!=record['manifest_sha256']:
            raise ValueError('预处理尚未完成或清单改变')
        for name,key in (('config.json','config_sha256'),('protocol.json','protocol_sha256')):
            if sha256(path/name)!=manifest[key]:raise ValueError('预处理配置或协议改变')
        data=read_json(path/'data-manifest.json')
        if sha256(path/'data-manifest.json')!=record['data_manifest_sha256'] or Path(data['root']).resolve()!=raw:
            raise ValueError('预处理原始来源不是新快照')
        if sha256(path/'summary.json')!=record['summary_sha256']:
            raise ValueError('预处理摘要改变')
    return {'solution_id':solution_id,'run_id':run_id,'data_snapshot_id':binding['snapshot_id'],
        'data_root':str(raw),'training_run':binding['training_run'],'holdout_run':binding['holdout_run'],
        'binding_path':str(binding_path),'binding_sha256':sha256(binding_path),
        'training_arguments':['--data-root',str(raw),'--training-run',binding['training_run'],
                              '--holdout-run',binding['holdout_run']],
        'resident_cache_argument':None,
        'cache_policy':'省略旧全局resident-cache参数，由本版加载器按新输入和新预处理身份构建缓存。',
        'evaluation_policy':'标签已修订，只与同快照已完成运行比较；首个结果记录为new_data_cohort_baseline，不重跑旧模型。',
        'evaluation_cohort_id':cohort['id'],'comparable_completed_run_ids':comparable,
        'first_cohort_baseline':not comparable,'design_parent_run_id':candidate.get('parent_run_id'),
        'design_parent_metrics_comparable':candidate.get('parent_run_id') in comparable,
        'dispatch_allowed':not refresh.get('block_new_training',True),
        'formal_training_performed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solution-id',required=True)
    args=parser.parse_args()
    print(json.dumps(resolve_new_run(ROOT,args.solution_id),ensure_ascii=False,indent=2))
