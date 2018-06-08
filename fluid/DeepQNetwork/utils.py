#-*- coding: utf-8 -*-
#File: utils.py
#Author: yobobobo(zhouboacmer@qq.com)
import paddle.fluid as fluid
import numpy as np
import tensorflow as tf

class Summary(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

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
