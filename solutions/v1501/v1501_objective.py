"""完整市场双头 Huber 与稳定化相关性，监督标签不改变股票上下文。"""

import torch
from torch.nn import functional as F


def stabilized_correlation(prediction, targets, epsilon):
    """沿股票维分别计算八个时点；分母平滑保证常数预测仍有有限梯度。"""
    if prediction.shape != targets.shape or prediction.ndim != 2 or prediction.shape[1] != 8:
        raise ValueError('完整日期的预测及目标须为 [股票, 8 时点]')
    valid = torch.isfinite(targets)
    count = valid.sum(dim=0)
    safe_count = count.clamp_min(1)
    scores = torch.where(valid, prediction.double(), 0)
    labels = torch.where(valid, targets.double(), 0)
    score_centered = torch.where(valid, scores - scores.sum(dim=0) / safe_count, 0)
    label_centered = torch.where(valid, labels - labels.sum(dim=0) / safe_count, 0)
    score_variance = score_centered.square().sum(dim=0) / safe_count
    label_variance = label_centered.square().sum(dim=0) / safe_count
    covariance = (score_centered * label_centered).sum(dim=0) / safe_count
    correlations = covariance / torch.sqrt((score_variance + epsilon ** 2) * (label_variance + epsilon ** 2))
    # 原训练目标已按协议过滤少标签或常数截面，此处只防御合成预检中的数值退化。
    eligible = (count >= 2) & (label_variance > 0)
    mean = correlations[eligible].mean() if eligible.any() else scores.sum() * 0
    std = score_variance[eligible].sqrt().mean() if eligible.any() else scores.sum().detach() * 0
    return mean, std.detach(), int(eligible.sum())



def full_day_objective(prediction, targets, sample_weights, spec):
    if prediction.shape != targets.shape or targets.ndim != 3 or targets.shape[1:] != (8,2):
        raise ValueError('完整市场要求 [股票日,8,2] 预测与目标')
    valid = torch.isfinite(targets[...,0])
    if not torch.equal(valid,torch.isfinite(targets[...,1])) or not valid.any():
        raise ValueError('双头监督须共享非空有效标签')
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('非有限市场预测')
    if sample_weights is None:
        sample_weights = torch.ones(len(targets),device=targets.device,dtype=torch.float64)
    if sample_weights.shape != (len(targets),) or not torch.isfinite(sample_weights).all() or (sample_weights<=0).any():
        raise ValueError('日期权重须为每股票日一个有限正数')
    if not torch.all(sample_weights==sample_weights[0]):
        raise ValueError('一次市场更新只允许一个完整日期')
    day_weight = sample_weights[0].double()
    main = F.huber_loss(prediction[...,0][valid].double(),targets[...,0][valid].double(),delta=spec['huber_delta'])
    rank = F.huber_loss(prediction[...,1][valid].double(),targets[...,1][valid].double(),delta=spec['auxiliary_rank_huber_delta'])
    correlation,std,sections = stabilized_correlation(prediction[...,0],targets[...,0],spec['correlation_std_epsilon'])
    loss = day_weight*(main+spec['auxiliary_rank_weight']*rank-spec['correlation_weight']*correlation)
    return loss, {'main_loss':day_weight*main.detach(),'rank_loss':day_weight*rank.detach(),
                  'correlation_loss':-day_weight*correlation.detach(),'stabilized_correlation':correlation.detach(),
                  'prediction_std':std,'day_weight':day_weight,'correlation_sections':sections,'rows':int(valid.sum())}
