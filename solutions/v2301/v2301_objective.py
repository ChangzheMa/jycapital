"""所有成员分别学习同一官方双目标，验证与推理使用固定均值。"""
import torch
from torch.nn import functional as F

STATISTICS=('ensemble_huber_loss','member_disagreement_rms','ensemble_prediction_std')


def combined_objective(members,targets,sample_weights,spec):
    if targets.ndim!=3 or targets.shape[1:]!=(8,2) or members.ndim!=4 or members.shape[:2]!=targets.shape[:2] or members.shape[-1]!=2:
        raise ValueError('成员输出须为[股票日,8,成员,2]，目标为[股票日,8,2]')
    if not torch.isfinite(members).all():raise FloatingPointError('非有限成员分数')
    valid=torch.isfinite(targets[...,0])
    if not torch.equal(valid,torch.isfinite(targets[...,1])) or not valid.any():raise ValueError('收益和排名目标须共享非空有效位置')
    if sample_weights is None:sample_weights=torch.ones(len(targets),dtype=torch.float64,device=targets.device)
    if sample_weights.shape!=(len(targets),) or not torch.isfinite(sample_weights).all() or (sample_weights<=0).any():
        raise ValueError('日期权重须为每个股票日一个有限正数')
    weights=sample_weights.double()[:,None].expand(-1,8)[valid];weights=weights/weights.sum()
    scores=members[valid].double();labels=targets[valid].double()
    main=F.huber_loss(scores[...,0],labels[:,None,0].expand_as(scores[...,0]),delta=spec['huber_delta'],reduction='none').mean(dim=-1)
    rank=F.huber_loss(scores[...,1],labels[:,None,1].expand_as(scores[...,1]),delta=spec['auxiliary_rank_huber_delta'],reduction='none').mean(dim=-1)
    main_loss=(weights*main).sum();rank_loss=(weights*rank).sum()
    loss=main_loss+spec['auxiliary_rank_weight']*rank_loss
    if not torch.isfinite(loss):raise FloatingPointError('非有限成员平均损失')
    prediction=members.mean(dim=-2)[valid].double()
    ensemble_huber=(weights*F.huber_loss(prediction[:,0],labels[:,0],delta=spec['huber_delta'],reduction='none')).sum()
    centered=scores.detach()[...,0]-scores.detach()[...,0].mean(dim=-1,keepdim=True)
    disagreement=(weights*centered.square().mean(dim=-1)).sum().sqrt()
    mean=(weights*prediction.detach()[:,0]).sum()
    spread=(weights*(prediction.detach()[:,0]-mean).square()).sum().sqrt()
    return loss,{'main_loss':main_loss.detach(),'rank_loss':rank_loss.detach(),
                 'ensemble_huber_loss':ensemble_huber.detach(),'member_disagreement_rms':disagreement,
                 'ensemble_prediction_std':spread,'rows':int(valid.sum())}
