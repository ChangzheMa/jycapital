"""两文件提交的 Notebook 入口模板；导入时不读取数据、不创建附件。"""

ARTIFACT_SHA256 = "__ARTIFACT_SHA256__"
_RUNTIME = None


def _bundle_directory():
    from pathlib import Path
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def _read_bundle():
    import hashlib
    import json
    raw = (_bundle_directory() / "v2-model.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARTIFACT_SHA256:
        raise ValueError("v2-model.json 与 Notebook 不匹配")
    return json.loads(raw.decode("utf-8"))


def _load_modules(bundle, paths):
    import torch
    import builtins
    import hashlib
    from pathlib import Path
    from types import ModuleType
    modules = {}

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    for relative in paths:
        entry = bundle["sources"][relative]
        if hashlib.sha256(entry["text"].encode("utf-8")).hexdigest() != entry["sha256"]:
            raise ValueError("内嵌源码哈希不匹配: " + relative)
        name = Path(relative).stem
        module = ModuleType(name)
        module.__file__ = str(_bundle_directory() / Path(relative).name)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=local_import)
        modules[name] = module
        exec(compile(entry["text"], module.__file__, "exec"), module.__dict__)
    return modules


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        bundle = _read_bundle()
        modules = _load_modules(bundle, bundle["runtime_sources"])
        _RUNTIME = bundle, modules
    return _RUNTIME


def main(datasources, start_date, end_date):
    bundle, modules = _runtime()
    return modules["v2_predict"].predict(
        datasources, start_date, end_date, model_dir=_bundle_directory(), bundle=bundle,
    )


def reproduce(workspace, data_root):
    """显式调用才启动预处理和训练；workspace 必须是不存在的新目录。"""
    bundle = _read_bundle()
    modules = _load_modules(bundle, ["solutions/v2/v2_reproduce.py"])
    return modules["v2_reproduce"].reproduce(bundle, workspace, data_root)
