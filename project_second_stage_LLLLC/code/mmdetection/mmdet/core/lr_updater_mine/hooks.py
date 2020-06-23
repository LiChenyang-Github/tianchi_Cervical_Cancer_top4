"""
    @lichenyang: 2020.01.11

    The customized hooks for lr update.
"""

from __future__ import division

from mmcv.runner.hooks import LrUpdaterHook
from math import cos, pi


class CosineLrUpdaterHookMine(LrUpdaterHook):
    """
        Use my own max_epochs instead of the max_epochs get from runner.

        Because the runner.max_epochs is from cfg.total_epochs.
        In some case, we don't want the max_progress is equal to cfg.total_epochs, 
        then we can use this func.
    """

    def __init__(self, target_lr=0, max_epochs=None, max_iters=None, **kwargs):
        self.target_lr = target_lr
        self.max_epochs = max_epochs
        self.max_iters = max_iters
        super(CosineLrUpdaterHookMine, self).__init__(**kwargs)
        
        # assert (self.max_epochs is not None) or (self.max_iters is not None)
        print("Use customized lr updater {}".format(self.__class__.__name__))
        if self.by_epoch:
            assert self.max_epochs is not None
            print("max_epochs for lr policy is {}".format(self.max_epochs))
        else:
            assert self.max_iters is not None
            print("max_iters for lr policy is {}".format(self.max_iters))


    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = self.max_epochs
        else:
            progress = runner.iter
            max_progress = self.max_iters
        return self.target_lr + 0.5 * (base_lr - self.target_lr) * \
            (1 + cos(pi * (progress / max_progress)))
