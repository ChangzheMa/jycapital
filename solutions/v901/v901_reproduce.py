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


def reproduce(bundle, workspace, data_root, *, training_confirmed=False):
    if not training_confirmed:
        raise ValueError("v901 训练复现尚待用户明确确认")
    workspace, data = Path(workspace).resolve(), Path(data_root).resolve()
    if not data.is_dir():
        raise FileNotFoundError(data)
    project = restore_sources(bundle, workspace)
    solution = project / "solutions/v901"
    training, holdout, run = (workspace / name for name in ("preprocessing-train", "preprocessing-holdout", "run"))
    commands = [
        ["v901_prepare.py", "--run-dir", str(training), "--data-root", str(data)],
        ["v901_apply.py", "--training-run", str(training), "--run-dir", str(holdout), "--data-root", str(data)],
        ["v901_train.py", "--training-run", str(training), "--holdout-run", str(holdout), "--run-dir", str(run), "--protocol", str(project / "evaluation/protocols/local-v2-ic.json"), "--data-root", str(data), "--training-confirmed", "--reproduction-only"],
    ]
    for script, *arguments in commands:
        subprocess.run([sys.executable, str(solution / script), *arguments], cwd=project, check=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-confirmed", action="store_true")
    args = parser.parse_args()
    if not args.training_confirmed:
        parser.error("v901 训练复现尚待用户确认；确认后才可传入 --training-confirmed")
    reproduce(json.loads(args.artifact.read_text(encoding="utf-8")), args.workspace,
              args.data_root, training_confirmed=True)
