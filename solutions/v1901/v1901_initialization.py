"""本地父版 EMA 初始化、独立复现载荷和两组学习率。"""
from pathlib import Path

import torch

from artifacts import log_event, sha256, write_json
from v1901_text_artifact import state_digest


def initialize_from_parent(model, config, root, run, initialization_state=None):
    spec = config["initialization"]
    if initialization_state is None:
        source = Path(root) / "outputs/runs" / spec["parent_run_id"] / "best.pt"
        if sha256(source) != spec["model_sha256"]:
            raise ValueError("父版最佳 EMA checkpoint 与冻结指纹不同")
        saved = torch.load(source, map_location="cpu", weights_only=True)
        if saved.get("weight_source") != "ema":
            raise ValueError("本方案只接受冻结父版的 EMA 初始化")
        payload = {"state_dict": saved["state_dict"], "parent_config": saved["config"],
                   "parent_run_id": spec["parent_run_id"], "parent_model_sha256": spec["model_sha256"]}
    else:
        source = Path(initialization_state)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if payload.get("parent_run_id") != spec["parent_run_id"] or payload.get("parent_model_sha256") != spec["model_sha256"]:
            raise ValueError("独立复现初始化载荷的父版身份不一致")
    if payload["parent_config"]["data"] != config["data"]:
        raise ValueError("父版输入表达与本版不一致")
    actual = state_digest(payload["state_dict"])
    if actual != spec["state_sha256"]:
        raise ValueError("初始化张量指纹不一致")
    model.encoder.load_state_dict(payload["state_dict"], strict=True)
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("本方案要求所有参数继续训练")
    target = Path(run) / "initialization-state.pt"
    torch.save(payload, target)
    record = {"method": "local_parent_ema", "parent_run_id": spec["parent_run_id"],
              "parent_model_sha256": spec["model_sha256"], "parent_state_sha256": actual,
              "input_contract_unchanged": True, "all_parameters_trainable": True,
              "payload": target.name, "payload_sha256": sha256(target),
              "parent_tensor_count": len(payload["state_dict"]), "source": str(source.resolve())}
    write_json(Path(run) / "initialization.json", record)
    log_event(run, "parent_ema_initialized", **record)
    return record


def create_optimizer(model, spec):
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    new = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    return torch.optim.AdamW([
        {"params": encoder, "lr": spec["backbone_learning_rate"], "role": "parent_encoder"},
        {"params": new, "lr": spec["learning_rate"], "role": "conditional_factor_modules"},
    ], weight_decay=spec["weight_decay"])
