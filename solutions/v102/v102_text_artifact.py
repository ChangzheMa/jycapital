"""用 JSON 数值数组无损保存模型张量，不使用二进制包装或精度截断。"""

import hashlib
import json
from pathlib import Path

import torch

FORMAT = "jycapital-v102-text-v1"
DTYPES = {name: getattr(torch, name) for name in ("float32", "float64", "int32", "int64", "bool")}


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def encode_state(state):
    result = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype not in DTYPES or not torch.isfinite(value).all():
            raise ValueError("不支持的张量类型或非有限权重: " + name)
        result[name] = {"dtype": dtype, "shape": list(value.shape), "values": value.reshape(-1).tolist()}
    return result


def decode_state(bundle):
    if bundle["format"] != FORMAT:
        raise ValueError("未知的文本模型格式")
    state = {}
    for name, item in bundle["state_dict"].items():
        if item["dtype"] not in DTYPES:
            raise ValueError("未知张量类型: " + name)
        value = torch.tensor(item["values"], dtype=DTYPES[item["dtype"]]).reshape(item["shape"])
        if not torch.isfinite(value).all():
            raise ValueError("非有限权重: " + name)
        state[name] = value
    return state


def read_bundle(path):
    bundle = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if bundle["format"] != FORMAT:
        raise ValueError("未知的文本模型格式")
    return bundle
