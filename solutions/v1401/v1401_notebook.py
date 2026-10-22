"""三文件冻结组合入口；源码和配置由导出器填入，不在导入时查询行情。"""

_CONFIG = None  # EXPORT_CONFIG
_OWN_SOURCES = None  # EXPORT_SOURCES
_RUNTIME = None


def _bundle_directory():
    from pathlib import Path
    return Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()


def _read_bundles():
    import json
    return [json.loads((_bundle_directory() / name).read_bytes().decode('utf-8-sig'))
            for name in ('v1002-model.json', 'v1101-model.json')]


def _load_modules(sources, paths):
    import torch
    import builtins
    from pathlib import Path
    from types import ModuleType
    modules = {}

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    for relative in paths:
        module = ModuleType(Path(relative).stem)
        module.__file__ = str(_bundle_directory() / Path(relative).name)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
        modules[module.__name__] = module
        exec(compile(sources[relative]['text'], module.__file__, 'exec'), module.__dict__)
    return modules


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        bundles = _read_bundles()
        sources = {name: entry for bundle in bundles for name, entry in bundle['sources'].items()}
        sources.update(_OWN_SOURCES)
        paths = [name for bundle in bundles for name in bundle['runtime_sources']] + list(_OWN_SOURCES)
        _RUNTIME = bundles, _load_modules(sources, paths)
    return _RUNTIME


def main(datasources, start_date, end_date):
    bundles, modules = _runtime()
    return modules['v1401_predict'].predict(datasources, start_date, end_date,
                                           bundles=bundles, ensemble_config=_CONFIG)


def reproduce(workspace, data_root, *, training_confirmed=False):
    """显式复现两个组件，串行运行；组合函数见 combine_predictions。"""
    from pathlib import Path
    if not training_confirmed:
        raise ValueError('组件训练复现须显式确认')
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise FileExistsError(workspace)
    workspace.mkdir(parents=True)
    results = {}
    for bundle in _read_bundles():
        version = bundle['config']['solution_id']
        name = version + '_reproduce'
        modules = _load_modules(bundle['sources'], [f'solutions/{version}/{name}.py'])
        results[version] = modules[name].reproduce(bundle, workspace / version, data_root, training_confirmed=True)
    return results


def combine_predictions(v1002_predictions, v1101_predictions):
    """组合两个组件的完整同截面预测，可用于复现后评估。"""
    _, modules = _runtime()
    combine = modules['v1401_combine']
    keys, scores = combine.align_components(v1002_predictions, v1101_predictions)
    return combine.combine_scores(keys, scores)
