#-*- coding: utf-8 -*-
#File: utils.py

import paddle.fluid as fluid
import numpy as np


def fluid_argmax(x):
    """
    Get index of max value for the last dimension
    """
    _, max_index = fluid.layers.topk(x, k=1)
    return max_index


def fluid_flatten(x):
    """
    Flatten fluid variable along the first dimension
    """
    return fluid.layers.reshape(x, shape=[-1, np.prod(x.shape[1:])])
