"""仅允许已审核的 v201/v301 验证加载器升级，保留完整训练状态。"""

import ast
import hashlib
import os
from pathlib import Path
import shutil

import psutil

from artifacts import read_json, write_json, sha256, snapshot_sources, now

# 精确限定本次迁移前后的训练入口 AST；其他版本须重新审查，不能跳过源码校验。
TRAINERS = {
    "v201": ("1744f465ac5cd278b4e002a9992ec2dfa4f4f21924e21a83609fe464df7c048f", "08ba0bb7afd518698ac39272b87f4ddd1ba99616517b3995836f243a672e818d"),
    "v301": ("6fc9033e49c55378f211f01f5737501a420edac7b767736e27ced2254df5b771", "f520ab71c336b5c32d21ce8fc03aa56f2c1fbddd6407c7da10de0ed908af93d9"),
}


def ast_digest(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def validate_upgrade(project, run, manifest, old_source=None):
    project, run = Path(project), Path(run)
    version = manifest["solution_id"]
    if version not in TRAINERS or manifest["status"] == "complete":
        raise ValueError("只支持未完成 v201/v301 运行的验证加载器升级")
    config = read_json(run / "config.json")
    if config["training"]["batching"] != "global_shuffled_samples":
        raise ValueError("验证优化升级不转换月内／全局训练模式")
    if not (run / "last.pt").is_file():
        raise ValueError("升级需要可恢复的 last.pt")
    old_source = Path(old_source) if old_source is not None else run / "source"
    trainer = f"solutions/{version}/{version}_train.py"
    before, after = TRAINERS[version]
    if ast_digest(old_source / trainer) != before or ast_digest(project / trainer) != after:
        raise ValueError("训练入口不属于已审核的验证优化升级版本")
    for relative, expected in manifest["source"]["files"].items():
        if sha256(old_source / relative) != expected:
            raise ValueError("原源码快照校验失败: " + relative)
        if relative != trainer and sha256(project / relative) != expected:
            raise ValueError("验证升级不能同时修改其他已有源码: " + relative)
    return trainer


def ensure_no_other_trainer(run, version):
    ignored = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid in ignored:
            continue
        args = process.info["cmdline"] or []
        if not any(Path(arg).name == version + "_train.py" for arg in args) or "--run-dir" not in args:
            continue
        target = args[args.index("--run-dir") + 1]
        if Path(target).resolve() == Path(run).resolve():
            raise RuntimeError("同一 run 的训练进程仍在运行，必须先停止后再升级恢复")


def upgrade_evaluation_sources(project, run, manifest):
    """原目录内升级执行路径；模型、AdamW、RNG、游标和早停 checkpoint 不改写。"""
    project, run = Path(project), Path(run)
    ensure_no_other_trainer(run, manifest["solution_id"])
    archive = run / "source-upgrades/evaluation-cache-v1"
    if archive.exists():
        record = read_json(archive / "upgrade.json")
        if record["status"] == "complete":
            # 已完成升级时允许保留 CLI 标记，但不能借此放行后续源码变化。
            for path, digest in manifest["source"]["files"].items():
                if sha256(project / path) != digest:
                    raise ValueError("已升级运行的源码再次改变: " + path)
            return manifest
        original = read_json(archive / "manifest-before.json")
        validate_upgrade(project, run, original, archive / "source")
    else:
        validate_upgrade(project, run, manifest)
        archive.mkdir(parents=True)
        write_json(archive / "manifest-before.json", manifest)
        shutil.copytree(run / "source", archive / "source")
        record = {"kind": "evaluation_cache_v1", "status": "prepared", "created_at": now(),
                  "preserved": {name: sha256(run / name) for name in (
                      "last.pt", "config.json", "protocol.json", "structural-sections.json", "data-manifest.json")}}
        if (run / "best.pt").exists():
            record["preserved"]["best.pt"] = sha256(run / "best.pt")
        write_json(archive / "upgrade.json", record)
    for name, expected in record["preserved"].items():
        if sha256(run / name) != expected:
            raise ValueError("升级期间训练状态改变: " + name)
    manifest["source"] = snapshot_sources(project, run, solution_id=manifest["solution_id"])
    manifest["evaluation_upgrade"] = {"kind": "evaluation_cache_v1", "at": now(),
                                      "record": "source-upgrades/evaluation-cache-v1/upgrade.json"}
    write_json(run / "manifest.json", manifest)
    record.update(status="complete", completed_at=now(), source_after=manifest["source"])
    write_json(archive / "upgrade.json", record)
    return manifest
