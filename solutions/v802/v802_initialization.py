"""从本地父模型或其文本初始张量恢复主干，之后允许更新。"""

from pathlib import Path

import torch

from artifacts import read_json, sha256
from v802_text_artifact import state_digest, decode_state

ROOT = Path(__file__).resolve().parents[2]


def initialize_backbone(model, config, artifact=None):
    spec = config["initialization"]
    if artifact is None:
        path = ROOT / "outputs/runs" / spec["run_id"] / "best.pt"
        if sha256(path) != spec["model_sha256"]:
            raise ValueError("初始化父模型文件指纹不符")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("weight_source") != "ema" or checkpoint["config"]["data"] != config["data"]:
            raise ValueError("初始化要求本地父模型 EMA 与同一输入配置")
        prefix = spec["state_prefix"]
        state = {name.removeprefix(prefix): value for name, value in checkpoint["state_dict"].items()
                 if name.startswith(prefix)}
    else:
        state = decode_state(read_json(artifact))
    if state_digest(state) != spec["state_dict_sha256"]:
        raise ValueError("初始化主干张量指纹不符")
    model.backbone.load_state_dict(state, strict=True)
    model.backbone.requires_grad_(True).train(model.training)
    return {name: value.detach().cpu().clone() for name, value in state.items()}
