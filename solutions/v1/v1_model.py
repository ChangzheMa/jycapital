"""小型 MLP；PyTorch 训练、NumPy 推理使用同一组完整参数。"""

import json
from pathlib import Path

import numpy as np


def create_model(config):
    import torch.nn as nn

    size = len(config["data"]["fields"]) * config["data"]["window_bars"]
    modules = []
    for hidden in config["model"]["hidden_dims"]:
        modules.extend([nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(config["model"]["dropout"])])
        size = hidden
    modules.append(nn.Linear(size, 1))
    return nn.Sequential(*modules)


def export_model(model, config, path):
    import torch.nn as nn

    layers = [
        {"weight": layer.weight.detach().cpu().numpy().tolist(), "bias": layer.bias.detach().cpu().numpy().tolist()}
        for layer in model if isinstance(layer, nn.Linear)
    ]
    payload = {"format": "jycapital-mlp-v1", "config": config, "layers": layers}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class NumpyMLP:
    def __init__(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["format"] != "jycapital-mlp-v1":
            raise ValueError("未知模型格式")
        self.config = payload["config"]
        self.layers = [(np.asarray(layer["weight"], dtype=np.float32), np.asarray(layer["bias"], dtype=np.float32)) for layer in payload["layers"]]
        expected = [self.config["data"]["window_bars"] * len(self.config["data"]["fields"]), *self.config["model"]["hidden_dims"], 1]
        if len(self.layers) != len(expected) - 1:
            raise ValueError("权重层数与配置不一致")
        for i, (weight, bias) in enumerate(self.layers):
            if weight.shape != (expected[i + 1], expected[i]) or bias.shape != (expected[i + 1],):
                raise ValueError("权重形状与配置不一致")
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError("权重必须是有限值")

    def predict(self, x):
        output = np.empty(len(x), dtype=np.float64)
        batch = self.config["inference"]["batch_size"]
        for start in range(0, len(x), batch):
            value = np.asarray(x[start:start + batch], dtype=np.float32)
            for i, (weight, bias) in enumerate(self.layers):
                value = value @ weight.T + bias
                if i + 1 < len(self.layers):
                    np.maximum(value, 0, out=value)
            output[start:start + len(value)] = value[:, 0] / self.config["training"]["target_scale"]
        if not np.isfinite(output).all():
            raise ValueError("模型产生非有限预测")
        return output
