"""
hinge loss
"""

import sys

sys.path.append("../../../")
import models.matching.paddle_layers as layers


class HingeLoss(object):
    """
    Hing Loss Calculate class
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.margin = conf_dict["loss"]["margin"]

    def compute(self, pos, neg):
        """
        compute loss
        """
        elementwise_max = layers.ElementwiseMaxLayer()
        elementwise_add = layers.ElementwiseAddLayer()
        elementwise_sub = layers.ElementwiseSubLayer()
        constant = layers.ConstantLayer()
        reduce_mean = layers.ReduceMeanLayer()
        loss = reduce_mean.ops(
            elementwise_max.ops(
                constant.ops(neg, neg.shape, "float32", 0.0),
                elementwise_add.ops(
                    elementwise_sub.ops(neg, pos),
                    constant.ops(neg, neg.shape, "float32", self.margin))))
        return loss
