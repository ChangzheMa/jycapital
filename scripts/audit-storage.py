"""只读统计仓库文件长度；跳过所有目录联接和符号链接，不删除文件。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import stat


def inventory(root: Path) -> dict:
    directories = defaultdict(lambda: {"bytes": 0, "files": 0})
    files, skipped, errors = [], [], []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = Path(entry.path).relative_to(root).as_posix()
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                        reparse = getattr(metadata, "st_file_attributes", 0) & getattr(
                            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
                        )
                        if reparse or stat.S_ISLNK(metadata.st_mode):
                            skipped.append(relative)
                        elif stat.S_ISDIR(metadata.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(metadata.st_mode):
                            files.append({"path": relative, "bytes": metadata.st_size,
                                          "mtime_ns": metadata.st_mtime_ns})
                            parent = Path(relative).parent
                            while True:
                                summary = directories[parent.as_posix()]
                                summary["bytes"] += metadata.st_size
                                summary["files"] += 1
                                if parent == Path("."):
                                    break
                                parent = parent.parent
                    except OSError as exc:
                        errors.append({"path": relative, "error": str(exc)})
        except OSError as exc:
            errors.append({"path": directory.relative_to(root).as_posix(), "error": str(exc)})
    usage = shutil.disk_usage(root)
    return {
        "schema_version": 1,
        "root": str(root),
        "generated_at": datetime.now().astimezone().isoformat(),
        "measurement": "文件逻辑长度；不等同于压缩、稀疏文件及硬链接的实际占用。扫描期间文件可能变化。",
        "disk": dict(zip(("total", "used", "free"), usage)),
        "directories": dict(directories),
        "files": files,
        "skipped_reparse_points_and_symlinks": skipped,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.root.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output), "disk": report["disk"],
        "top_level": {k: v for k, v in report["directories"].items() if "/" not in k},
        "skipped_links": len(report["skipped_reparse_points_and_symlinks"]),
        "errors": report["errors"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
