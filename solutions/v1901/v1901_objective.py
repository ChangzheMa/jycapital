"""完整截面直接收益监督及训练后验指导；正式预测只取历史先验。"""
import math

import torch
from torch.nn import functional as F


def stabilized_correlation(prediction, targets, epsilon):
    if prediction.shape != targets.shape or prediction.ndim != 2 or prediction.shape[1] != 8:
        raise ValueError("完整日期的预测及目标须为 [股票,8]")
    valid = torch.isfinite(targets)
    count = valid.sum(dim=0).clamp_min(1)
    scores = torch.where(valid, prediction.double(), 0)
    labels = torch.where(valid, targets.double(), 0)
    scores = torch.where(valid, scores - scores.sum(dim=0) / count, 0)
    labels = torch.where(valid, labels - labels.sum(dim=0) / count, 0)
    score_variance = scores.square().sum(dim=0) / count
    label_variance = labels.square().sum(dim=0) / count
    covariance = (scores * labels).sum(dim=0) / count
    correlation = covariance / ((score_variance + epsilon ** 2) * (label_variance + epsilon ** 2)).sqrt()
    eligible = (count >= 2) & (label_variance > 0)
    mean = correlation[eligible].mean() if eligible.any() else scores.sum() * 0
    std = score_variance[eligible].sqrt().mean() if eligible.any() else scores.detach().sum() * 0
    return mean, std.detach(), int(eligible.sum())


def mean_clocks(values, valid):
    count = valid.sum(dim=0)
    per_clock = torch.where(valid, values, 0).sum(dim=0) / count.clamp_min(1)
    return per_clock[count > 0].mean()


def full_day_objective(prior, posterior, targets, sample_weights, spec):
    prediction = prior["prediction"]
    if prediction.shape != targets.shape or targets.ndim != 3 or targets.shape[1:] != (8, 2):
        raise ValueError("完整市场要求 [股票日,8,2] 的预测和目标")
    valid = torch.isfinite(targets[..., 0])
    if not torch.equal(valid, torch.isfinite(targets[..., 1])) or not valid.any():
        raise ValueError("双头监督须共享非空有效标签")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("非有限历史先验预测")
    if sample_weights is None:
        sample_weights = torch.ones(len(targets), dtype=torch.float64, device=targets.device)
    if sample_weights.shape != (len(targets),) or not torch.isfinite(sample_weights).all() or (sample_weights <= 0).any():
        raise ValueError("日期权重必须有限且为正")
    if not torch.all(sample_weights == sample_weights[0]):
        raise ValueError("一次更新只允许一个完整日期")
    weight = sample_weights[0].double()
    labels = torch.where(valid[..., None], targets.double(), 0)
    # 直接收益误差不按模型预测的不确定性降权。
    main = .5 * mean_clocks((prediction[..., 0].double() - labels[..., 0]).square(), valid)
    rank = mean_clocks(F.huber_loss(prediction[..., 1].double(), labels[..., 1],
                                   delta=spec["auxiliary_rank_huber_delta"], reduction="none"), valid)
    correlation, std, sections = stabilized_correlation(prediction[..., 0], targets[..., 0], spec["correlation_std_epsilon"])
    loading = prior["loadings"][..., 0, :].double()
    q_mean, q_std = posterior["posterior_mean"].double(), posterior["posterior_std"].double()
    p_mean, p_std = prior["prior_mean"].double(), prior["prior_std"].double()
    posterior_mean = prior["individual"][..., 0].double() + torch.einsum("nck,ck->nc", loading, q_mean)
    factor_variance = torch.einsum("nck,ck->nc", loading.square(), q_std.square())
    noise_variance = prior["idiosyncratic_std"].double().square()
    expected_error = (labels[..., 0] - posterior_mean).square() + factor_variance
    nll = mean_clocks(.5 * (math.log(2 * math.pi) + noise_variance.log() + expected_error / noise_variance), valid)
    eligible = valid.any(dim=0)
    kl_by_factor = .5 * ((q_std.square() + (q_mean - p_mean).square()) / p_std.square()
                         - 1 + 2 * (p_std.log() - q_std.log()))
    kl = kl_by_factor[eligible].mean()
    loss = weight * (main + spec["auxiliary_rank_weight"] * rank - spec["correlation_weight"] * correlation
                     + spec["posterior_likelihood_weight"] * nll + spec["prior_kl_weight"] * kl)
    if not torch.isfinite(loss):
        raise FloatingPointError("非有限条件因子训练目标")
    posterior_correlation, _, _ = stabilized_correlation(posterior_mean.detach(), targets[..., 0], spec["correlation_std_epsilon"])
    return loss, {"main_loss": weight * main.detach(), "rank_loss": weight * rank.detach(),
        "correlation_loss": -weight * correlation.detach(), "stabilized_correlation": correlation.detach(),
        "prediction_std": std, "day_weight": weight,
        "posterior_nll": weight * nll.detach(), "prior_kl": weight * kl.detach(),
        "posterior_correlation": posterior_correlation.detach(),
        "mean_prior_std": p_std[eligible].mean().detach(), "mean_posterior_std": q_std[eligible].mean().detach(),
        "mean_loading_norm": loading[valid].square().sum(dim=-1).sqrt().mean().detach(),
        "correlation_sections": sections, "rows": int(valid.sum())}
