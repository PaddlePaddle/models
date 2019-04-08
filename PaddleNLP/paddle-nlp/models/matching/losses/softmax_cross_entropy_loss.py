"""
softmax loss
"""

import sys

sys.path.append("../../../")
import models.matching.paddle_layers as layers


class SoftmaxCrossEntropyLoss(object):
    """
    Softmax with Cross Entropy Loss Calculate
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        pass

    def compute(self, input, label):
        """
        compute loss
        """
        softmax_with_cross_entropy = layers.SoftmaxWithCrossEntropyLayer()
        reduce_mean = layers.ReduceMeanLayer()
        cost = softmax_with_cross_entropy.ops(input, label)
        avg_cost = reduce_mean.ops(cost)
        return avg_cost
