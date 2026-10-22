"""完整截面的条件潜在收益结构；后验只供训练，预测只调用历史先验。"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from v1901_backbone import StockEncoder, validate_model_config as validate_encoder_config
from v1901_schema import FREQUENCIES, prepare_model_inputs


def validate_model_config(config):
    validate_encoder_config(config)
    spec = config["model"]
    for key in ("encoding_stock_batch_size", "latent_factor_count", "latent_hidden_dim"):
        if type(spec.get(key)) is not int or spec[key] < 1:
            raise ValueError(key + " 必须为正整数")
    for key in ("latent_minimum_std", "idiosyncratic_minimum_std"):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(key + " 必须为 (0,1) 内的有限值")
    if spec.get("activation_checkpoint") is not True or spec.get("posterior_at_inference") is not False:
        raise ValueError("编码须使用重计算，后验不得参与推理")
    if spec.get("posterior_inputs") != "training_labels_only" or spec.get("market_membership") != "observed_universe_at_each_clock":
        raise ValueError("后验和市场成员信息边界不匹配")


class MultiFrequencyGRU(nn.Module):
    """GRU 行情编码器加条件因子解码器；共享状态只跨股票、不跨未来时点。"""

    def __init__(self, config):
        super().__init__()
        validate_model_config(config)
        spec = config["model"]
        self.encoder = StockEncoder(config)
        self.window_bars = dict(config["data"]["window_bars"])
        self.encoding_stock_batch_size = spec["encoding_stock_batch_size"]
        self.feature_width = spec["fusion_hidden_dim"]
        self.factor_count = spec["latent_factor_count"]
        self.minimum_std = float(spec["latent_minimum_std"])
        self.idiosyncratic_minimum_std = float(spec["idiosyncratic_minimum_std"])
        width, factors = spec["latent_hidden_dim"], self.factor_count
        self.project = nn.Sequential(nn.LayerNorm(self.feature_width), nn.Linear(self.feature_width, width), nn.GELU())
        self.keys = nn.Linear(width, width, bias=False)
        self.values = nn.Linear(width, width)
        self.queries = nn.Parameter(torch.empty(factors, width))
        nn.init.normal_(self.queries, std=.02)
        self.prior = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        # 先验均值从零开始，加载父权重后初始预测精确保留父模型。
        nn.init.zeros_(self.prior[-1].weight)
        nn.init.zeros_(self.prior[-1].bias)
        with torch.no_grad():
            self.prior[-1].bias[1] = math.log(math.expm1(1 - self.minimum_std))
        self.loadings = nn.Linear(width, 2 * factors)
        nn.init.normal_(self.loadings.weight, std=.02)
        nn.init.zeros_(self.loadings.bias)
        self.idiosyncratic = nn.Linear(width, 1)
        nn.init.zeros_(self.idiosyncratic.weight)
        nn.init.constant_(self.idiosyncratic.bias, math.log(math.expm1(1 - self.idiosyncratic_minimum_std)))
        self.posterior_portfolios = nn.Linear(width, factors)
        self.posterior = nn.Sequential(nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.normal_(self.posterior[-1].weight, std=.02)
        nn.init.zeros_(self.posterior[-1].bias)
        with torch.no_grad():
            self.posterior[-1].bias[1] = math.log(math.expm1(1 - self.minimum_std))

    def _encode_part(self, *inputs):
        windows = dict(zip(FREQUENCIES, inputs[:-1]))
        prediction, _, _, features = self.encoder.forward_group_components(windows, inputs[-1])
        return torch.cat([features.float(), prediction.float()], dim=-1)

    def encode_group(self, windows, latest_clock):
        if latest_clock.ndim != 2 or latest_clock.shape[1] != 8 or not len(latest_clock):
            raise ValueError("行情编码要求非空股票日与八维预测时钟")
        parts = []
        for start in range(0, len(latest_clock), self.encoding_stock_batch_size):
            stop = start + self.encoding_stock_batch_size
            values = [windows[f][start:stop] for f in FREQUENCIES] + [latest_clock[start:stop]]
            if self.training and torch.is_grad_enabled():
                part = checkpoint(self._encode_part, *values, use_reentrant=False, preserve_rng_state=True)
            else:
                part = self._encode_part(*values)
            parts.append(part)
        return torch.cat(parts, dim=0)

    def encode_reference(self, windows, clock):
        """独立逐时点编码，不复用八时点前缀，供针对性等价验证。"""
        windows, lengths = prepare_model_inputs(windows, clock, self.window_bars)
        vectors = [self.encoder.encoders[f](windows[f], lengths[f]) for f in FREQUENCIES]
        prediction, _, _, features = self.encoder.fuse_components(vectors, clock)
        return torch.cat([features.float(), prediction.float()], dim=-1)

    @staticmethod
    def membership(value, membership):
        if membership is None:
            return torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
        if membership.shape != value.shape[:2] or membership.dtype != torch.bool:
            raise ValueError("市场成员掩码须为 [股票,8] 布尔数组")
        return membership

    def decode_prior(self, encoded, membership=None):
        if encoded.ndim != 3 or encoded.shape[1:] != (8, self.feature_width + 2):
            raise ValueError("解码要求完整股票集合的八时点表示")
        membership = self.membership(encoded, membership)
        features, individual = encoded[..., :self.feature_width], encoded[..., self.feature_width:]
        hidden = self.project(features)
        keys, values = self.keys(hidden), self.values(hidden)
        logits = torch.einsum("ncd,kd->cnk", keys.float(), self.queries.float()) / math.sqrt(keys.shape[-1])
        present = membership.T[..., None]
        attention = torch.softmax(logits.masked_fill(~present, torch.finfo(logits.dtype).min), dim=1)
        attention = torch.where(present, attention, 0)
        context = torch.einsum("cnk,ncd->ckd", attention.to(values.dtype), values)
        distribution = self.prior(context).float()
        prior_mean = distribution[..., 0]
        prior_std = self.minimum_std + F.softplus(distribution[..., 1])
        loading = self.loadings(hidden).float().reshape(*hidden.shape[:2], 2, self.factor_count)
        prediction = individual.float() + torch.einsum("nchk,ck->nch", loading, prior_mean)
        prediction = torch.where(membership[..., None], prediction, 0)
        idiosyncratic_std = self.idiosyncratic_minimum_std + F.softplus(self.idiosyncratic(hidden).float()[..., 0])
        return {"prediction": prediction, "individual": individual.float(), "loadings": loading,
                "prior_mean": prior_mean, "prior_std": prior_std, "idiosyncratic_std": idiosyncratic_std,
                "hidden": hidden, "context": context, "membership": membership}

    def training_posterior(self, prior, targets):
        if not self.training:
            raise ValueError("后验仅用于训练，评估和推理必须调用历史先验")
        if targets.shape != prior["prediction"].shape:
            raise ValueError("后验标签须与完整截面双头目标形状一致")
        label = targets[..., 0].detach().float()
        valid = torch.isfinite(label)
        if (valid & ~prior["membership"]).any():
            raise ValueError("后验标签不得来自当时股票池之外")
        safe = torch.where(valid, label, 0)
        # 权重由行情表示学习；仅后验构造时按有效训练标签进行掩码。
        weights = torch.tanh(self.posterior_portfolios(prior["hidden"]).float())
        counts = valid.sum(dim=0).clamp_min(1)
        center = (weights * valid[..., None]).sum(dim=0) / counts[:, None]
        weights = torch.where(valid[..., None], weights - center, 0)
        weights = weights / weights.square().sum(dim=0, keepdim=True).clamp_min(1e-12).sqrt()
        portfolios = (weights * safe[..., None]).sum(dim=0)
        features = torch.cat([prior["context"].float(), portfolios[..., None]], dim=-1)
        distribution = self.posterior(features).float()
        return {"posterior_mean": distribution[..., 0],
                "posterior_std": self.minimum_std + F.softplus(distribution[..., 1])}

    def decode_group(self, encoded, membership=None):
        return self.decode_prior(encoded, membership)["prediction"]

    def forward_group(self, windows, latest_clock, membership=None):
        return self.decode_group(self.encode_group(windows, latest_clock), membership)

    def training_forward(self, windows, latest_clock, targets, membership=None):
        prior = self.decode_prior(self.encode_group(windows, latest_clock), membership)
        return prior, self.training_posterior(prior, targets)

    def forward_heads(self, windows, clock):
        if not torch.all(clock == clock[0]):
            raise ValueError("逐时点推理必须一次提供同一完整截面")
        index = int(clock[0].argmax())
        value = self.encode_reference(windows, clock)
        encoded = value.new_zeros(len(value), 8, value.shape[-1])
        encoded[:, index] = value
        present = torch.zeros(len(value), 8, dtype=torch.bool, device=value.device)
        present[:, index] = True
        return self.decode_group(encoded, present)[:, index]

    def forward(self, windows, clock):
        return self.forward_heads(windows, clock)[:, 0]
