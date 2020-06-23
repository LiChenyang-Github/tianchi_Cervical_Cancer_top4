import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    ###
    # txt_dir = "./output/statistic/bbox_head_loss/bbox_head_loss.txt"

    # for i in range(0, 7):
    #     loss_cur = loss[label==i]
    #     # print(loss_cur, len(loss_cur))
    #     if len(loss_cur) > 0:
    #         line_str = "{} {} {}\n".format(i, loss_cur.mean(), len(loss_cur))
    #         # print(line_str)
    #         with open(txt_dir, 'a') as f:
    #             f.write(line_str)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss




@LOSSES.register_module
class ArcFaceLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 s=32,
                 m=0.5,
                 eps=1e-7):
        super(ArcFaceLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

        # self.cls_criterion = cross_entropy

        self.s = s
        self.m = m
        self.eps = eps

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(cls_score.transpose(0, 1)[label]), -1.+self.eps, 1-self.eps)) + self.m)
        excl = torch.cat([torch.cat((cls_score[i, :y], cls_score[i, y+1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)

        loss_cls = -torch.mean(L)

        return self.loss_weight * loss_cls


@LOSSES.register_module
class AngularPenaltySMLoss(nn.Module):

    def __init__(self, loss_weight=1.0, \
            loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 32.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.loss_weight = loss_weight

    def forward(self, 
                cls_score, 
                labels,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        '''
        input shape (N, in_features)
        '''
        assert len(cls_score) == len(labels)
        assert torch.min(labels) >= 0

        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(cls_score.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(cls_score.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(cls_score.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((cls_score[i, :y], cls_score[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return self.loss_weight * (-torch.mean(L))