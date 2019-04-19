"""
log loss
"""

import sys

sys.path.append("../../../")
import models.matching.paddle_layers as layers


class LogLoss(object):
    """
    Log Loss Calculate
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        pass

    def compute(self, pos, neg):
        """
        compute loss
        """
        sigmoid = layers.SigmoidLayer()
        reduce_mean = layers.ReduceMeanLayer()
        loss = reduce_mean.ops(sigmoid.ops(neg - pos))
        return loss
