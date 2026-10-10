"""运行元数据、源码快照和文件完整性记录。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    names = sorted({dist.metadata["Name"] for dist in importlib.metadata.distributions() if dist.metadata.get("Name")})
    packages = {name: importlib.metadata.version(name) for name in names}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "packages": packages}


def snapshot_sources(project, run):
    project, run = Path(project), Path(run)
    paths = []
    for directory in ["solutions/v1", "evaluation", "common", "scripts", "tests"]:
        if (project / directory).exists():
            paths.extend(p for p in (project / directory).rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".txt"})
    sources = {}
    for path in sorted(paths):
        relative = path.relative_to(project)
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[relative.as_posix()] = sha256(target)
    local = project / "dai.py"
    if local.exists():
        target = run / "local-only" / "dai.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
    result = {"files": sources, "local_dai_sha256": sha256(local) if local.exists() else None}
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project, text=True).strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        result.update(git_commit=None, git_dirty=None)
    return result


def log_event(run, event, **fields):
    row = {"at": now(), "event": event, **fields}
    with (Path(run) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)
