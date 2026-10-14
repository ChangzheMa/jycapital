"""保留两空格对象缩进，将数值数组写为紧凑 JSON，不截断精度。"""

import json
from pathlib import Path


def compact_numeric_json(value, level=0):
    indent = "  " * level
    child = indent + "  "
    if isinstance(value, dict):
        if not value:
            return "{}"
        entries = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON 对象键必须是字符串")
            entries.append(child + json.dumps(key, ensure_ascii=False) + ": "
                           + compact_numeric_json(item, level + 1))
        return "{\n" + ",\n".join(entries) + "\n" + indent + "}"
    if isinstance(value, list):
        if all(isinstance(item, (int, float)) for item in value):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        entries = [child + compact_numeric_json(item, level + 1) for item in value]
        return "[\n" + ",\n".join(entries) + "\n" + indent + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_compact_numeric_json(path, value, *, max_bytes=None):
    content = (compact_numeric_json(value) + "\n").encode("utf-8")
    if max_bytes is not None and len(content) > max_bytes:
        raise ValueError(f"JSON 附件为 {len(content)} 字节，超过 {max_bytes} 字节上限")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return len(content)
