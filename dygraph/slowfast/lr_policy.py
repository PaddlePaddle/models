"""Learning rate policy."""

import math


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    #"""
    warmup_epochs = cfg.warmup_epochs  #34
    warmup_start_lr = cfg.warmup_start_lr  #0.01
    lr = lr_func_cosine(cur_epoch, cfg)

    # Perform warm up.
    if cur_epoch < warmup_epochs:
        lr_start = warmup_start_lr
        lr_end = lr_func_cosine(warmup_epochs, cfg)
        alpha = (lr_end - lr_start) / warmup_epochs
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_cosine(cur_epoch, cfg):
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    base_lr = cfg.base_lr  #0.1
    max_epoch = cfg.epoch  #196
    return base_lr * (math.cos(math.pi * cur_epoch / max_epoch) + 1.0) * 0.5
