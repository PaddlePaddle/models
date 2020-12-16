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
    warmup_epochs = cfg.OPTIMIZER.warmup_epochs  #34
    warmup_start_lr = cfg.OPTIMIZER.warmup_start_lr  #0.01
    if cfg.OPTIMIZER.lr_policy == "cosine":
        lr = lr_func_cosine(cur_epoch, cfg)
        lr_end = lr_func_cosine(warmup_epochs, cfg)
    elif cfg.OPTIMIZER.lr_policy == "steps_with_relative_lrs":
        lr = lr_func_steps_with_relative_lrs(cur_epoch, cfg)
        lr_end = lr_func_steps_with_relative_lrs(warmup_epochs, cfg)

    # Perform warm up.
    if cur_epoch < warmup_epochs:
        lr_start = warmup_start_lr
        #        lr_end = lr_func_cosine(warmup_epochs, cfg)
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
    base_lr = cfg.OPTIMIZER.base_lr  #0.1
    max_epoch = cfg.OPTIMIZER.max_epoch  #196
    return base_lr * (math.cos(math.pi * cur_epoch / max_epoch) + 1.0) * 0.5


def lr_func_steps_with_relative_lrs(cur_epoch, cfg):
    """
Retrieve the learning rate to specified values at specified epoch with the
steps with relative learning rate schedule.
Args:
    cfg (CfgNode): configs.
    cur_epoch (float): the number of epoch of the current training stage.
"""
    # get step index
    steps = cfg.OPTIMIZER.steps + [cfg.OPTIMIZER.max_epoch]
    for ind, step in enumerate(steps):
        if cur_epoch < step:
            break

    return cfg.OPTIMIZER.lrs[ind - 1] * cfg.OPTIMIZER.base_lr
