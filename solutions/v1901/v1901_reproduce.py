"""从文本提交恢复完整训练源码，显式运行预处理及一次正式训练。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def restore_sources(bundle, workspace):
    workspace = Path(workspace).resolve()
    project = workspace / "source"
    files = {}
    for relative in bundle["training_sources"]:
        target = (project / relative).resolve()
        if not target.is_relative_to(project) or target == project:
            raise ValueError("非法源码路径")
        entry = bundle["sources"][relative]
        content = entry["text"].encode("utf-8")
        files[target] = content
    workspace.mkdir(parents=True, exist_ok=False)
    for target, content in files.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return project


def reproduce(bundle, workspace, data_root, *, training_confirmed=False, initialization_bundle=None):
    if not training_confirmed:
        raise ValueError("v1901 训练复现尚待用户明确确认")
    workspace, data = Path(workspace).resolve(), Path(data_root).resolve()
    if not data.is_dir():
        raise FileNotFoundError(data)
    project = restore_sources(bundle, workspace)
    solution = project / "solutions/v1901"
    # 父权重是本地已训练初始化的一部分，独立复现须由文本载荷显式提供。
    sys.path.insert(0, str(solution))
    import torch
    from v1901_text_artifact import decode_state, state_digest
    initial = initialization_bundle if initialization_bundle is not None else bundle.get("initialization_state")
    if initial is None:
        raise ValueError("独立复现材料缺少冻结父版初始化的文本张量")
    initial_state = decode_state(initial)
    if state_digest(initial_state) != bundle["config"]["initialization"]["state_sha256"]:
        raise ValueError("独立复现父版初始化张量与配置不一致")
    initial_path = workspace / "initialization-state.pt"
    torch.save({"state_dict": initial_state, "parent_config": initial["parent_config"],
                "parent_run_id": initial["parent_run_id"],
                "parent_model_sha256": initial["parent_model_sha256"]}, initial_path)
    training, holdout, run = (workspace / name for name in ("preprocessing-train", "preprocessing-holdout", "run"))
    commands = [
        ["v1901_prepare.py", "--run-dir", str(training), "--data-root", str(data)],
        ["v1901_apply.py", "--training-run", str(training), "--run-dir", str(holdout), "--data-root", str(data)],
        ["v1901_train.py", "--training-run", str(training), "--holdout-run", str(holdout), "--run-dir", str(run), "--protocol", str(project / "evaluation/protocols/local-v2-ic.json"), "--data-root", str(data), "--training-confirmed", "--reproduction-only", "--initialization-state", str(initial_path)],
    ]
    for script, *arguments in commands:
        subprocess.run([sys.executable, str(solution / script), *arguments], cwd=project, check=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--initialization-artifact", type=Path)
    parser.add_argument("--training-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.training_confirmed:
        parser.error("v1901 训练复现尚待用户确认；确认后才可传入 --training-confirmed")
    bundle = json.loads(args.artifact.read_text(encoding="utf-8"))
    initialization = args.initialization_artifact
    if initialization is None and bundle.get("initialization_file"):
        initialization = args.artifact.parent / bundle["initialization_file"]
    initial = json.loads(initialization.read_text(encoding="utf-8")) if initialization is not None else None
    reproduce(bundle, args.workspace, args.data_root, training_confirmed=True, initialization_bundle=initial)
