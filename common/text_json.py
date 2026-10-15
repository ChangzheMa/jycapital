"""保留两空格对象缩进，将数值数组写为紧凑 JSON，不截断精度。"""

import json
import math
from pathlib import Path


class Float32Values(list):
    """显式标记 FP32 数值数组，以九位有效数字保存其可恢复精度。"""


def float32_number(value):
    if not math.isfinite(value):
        raise ValueError("FP32 数值必须有限")
    token = format(value, ".9g")
    # JSON 整数 -0 会被解析为正零，浮点负零须显式保留。
    return token + ".0" if token in ("0", "-0") else token


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
    if isinstance(value, Float32Values):
        return "[" + ",".join(float32_number(item) for item in value) + "]"
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
