"""将平铺提交材料还原为训练源码目录，顺序执行预处理和一次训练。"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="不存在的新目录，保存还原源码及全部实验产物")
    parser.add_argument("--data-root", type=Path, required=True, help="已授权的原始 Parquet 数据目录")
    args = parser.parse_args()
    package, workspace, data = Path(__file__).resolve().parent, args.workspace.resolve(), args.data_root.resolve()
    if not data.is_dir():
        raise FileNotFoundError(data)
    workspace.mkdir(parents=True, exist_ok=False)
    project = workspace / "source"
    solution = project / "solutions/v2"
    solution.mkdir(parents=True)
    for name in ("v2_preprocessing.py", "v2_prepare.py", "v2_apply.py", "v2_data.py", "v2_model.py", "v2_train.py", "v2_predict.py"):
        shutil.copyfile(package / name, solution / name)
    for source, target in (("preprocessing-config.json", "config.json"), ("model-config.json", "model-config.json"),
                           ("requirements.txt", "requirements.txt"), ("requirements-preprocessing.txt", "requirements-preprocessing.txt"),
                           ("requirements-training.txt", "requirements-training.txt")):
        shutil.copyfile(package / source, solution / target)
    for source, target in (("artifacts.py", "common/artifacts.py"), ("baseline_evaluation.py", "evaluation/baseline_evaluation.py"),
                           ("protocol.json", "evaluation/protocols/local-v1.json")):
        destination = project / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / source, destination)
    training, holdout, run = (workspace / name for name in ("preprocessing-train", "preprocessing-holdout", "run"))
    commands = [
        ["v2_prepare.py", "--run-dir", str(training), "--data-root", str(data)],
        ["v2_apply.py", "--training-run", str(training), "--run-dir", str(holdout), "--data-root", str(data)],
        ["v2_train.py", "--training-run", str(training), "--holdout-run", str(holdout), "--run-dir", str(run), "--data-root", str(data)],
    ]
    for script, *arguments in commands:
        subprocess.run([sys.executable, str(solution / script), *arguments], cwd=project, check=True)


if __name__ == "__main__":
    main()
