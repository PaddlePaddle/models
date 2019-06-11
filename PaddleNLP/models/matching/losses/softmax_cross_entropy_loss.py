"""
softmax loss
"""

import sys
import paddle.fluid as fluid

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
        reduce_mean = layers.ReduceMeanLayer()
        cost = fluid.layers.cross_entropy(input=input, label=label)
        avg_cost = reduce_mean.ops(cost)
        return avg_cost
