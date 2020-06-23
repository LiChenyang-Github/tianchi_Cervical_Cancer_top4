import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss

import torch


def one_hot_embedding(labels, num_classes=7):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


def py_sigmoid_focal_loss_mine(pred,
                              target,
                              weight=None,
                              gamma=2.0,
                              alpha=0.25,
                              reduction='mean',
                              avg_factor=None):
    
    t = one_hot_embedding(target)
    t = t[:,1:]
    t = t.cuda()

    p = pred.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha

    w = w * (1-pt).pow(gamma)

    loss = F.binary_cross_entropy_with_logits(pred, t, w.detach(), reduction='none')

    if weight is not None:
        weight = weight.view(-1, 1)

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss



# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            # loss_cls = self.loss_weight * sigmoid_focal_loss(
            #     pred,
            #     target,
            #     weight,
            #     gamma=self.gamma,
            #     alpha=self.alpha,
            #     reduction=reduction,
            #     avg_factor=avg_factor)
            # loss_cls = self.loss_weight * py_sigmoid_focal_loss(
            #     pred,
            #     target,
            #     weight,
            #     gamma=self.gamma,
            #     alpha=self.alpha,
            #     reduction=reduction,
            #     avg_factor=avg_factor)
            loss_cls = self.loss_weight * py_sigmoid_focal_loss_mine(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


@LOSSES.register_module
class SoftFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 num_classes=2):
        super(SoftFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = num_classes


    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * py_soft_sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
                num_classes=self.num_classes)
        else:
            raise NotImplementedError
        return loss_cls


def py_soft_sigmoid_focal_loss(pred,
                              target,
                              weight=None,
                              gamma=2.0,
                              alpha=0.25,
                              reduction='mean',
                              avg_factor=None,
                              num_classes=2):
    
    assert target.dim() == 2
    assert pred.shape[0] == target.shape[0]

    anchor_num = target.shape[0]

    gt_labels = target[:, -1].long()

    t = one_hot_embedding(gt_labels, num_classes)
    t = t[:,1:]
    t = t.cuda()

    p = pred.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha

    w = w * (1-pt).pow(gamma)

    loss = F.binary_cross_entropy_with_logits(pred, t, w.detach(), reduction='none')

    if weight is not None:
        weight = weight.view(-1, 1)
    else:
        weight = torch.ones((anchor_num, 1))

    weight *= torch.unsqueeze(target[torch.arange(anchor_num).long(), gt_labels], -1)


    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss