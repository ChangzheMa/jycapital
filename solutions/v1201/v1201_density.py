"""训练专用 Student-t 混合似然；推理模型不依赖本模块。"""

import math

import torch
from torch.nn import functional as F


def component_log_prob(means, raw_scales, targets, density):
    """最后一维为分支，targets 不含分支维；均在截面标准化目标单位下。"""
    if means.shape != raw_scales.shape or targets.shape != means.shape[:-1]:
        raise ValueError("Student-t 位置、尺度与目标的形状不匹配")
    degrees = density["degrees_of_freedom"]
    scales = density["minimum_scale"] + F.softplus(raw_scales.double())
    residual = (targets.double()[..., None] - means.double()) / scales
    constant = (math.lgamma((degrees + 1) / 2) - math.lgamma(degrees / 2)
                - 0.5 * math.log(degrees * math.pi))
    log_prob = constant - scales.log() - (degrees + 1) / 2 * torch.log1p(residual.square() / degrees)
    return log_prob, scales


def mixture_nll(component_log_probability, logits):
    if component_log_probability.shape != logits.shape:
        raise ValueError("混合分布与路由形状不匹配")
    return -torch.logsumexp(F.log_softmax(logits.double(), dim=-1) + component_log_probability, dim=-1)


def mixture_standard_deviation(means, scales, probabilities, degrees_of_freedom):
    """中心化方差公式，避免两个大二阶矩相减；不表示原始收益风险校准。"""
    mean = (probabilities * means).sum(dim=-1, keepdim=True)
    variance = (probabilities * (degrees_of_freedom / (degrees_of_freedom - 2) * scales.square()
                                + (means - mean).square())).sum(dim=-1)
    return variance.sqrt()
