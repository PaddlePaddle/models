from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle.fluid as fluid

class SoftmaxLoss():
    def __init__(self, class_dim):
        self.class_dim = class_dim

    def loss(self, input, label):
        out = self.fc_product(input, self.class_dim)
        loss = fluid.layers.cross_entropy(input=out, label=label)
        return loss, out

    def fc_product(self, input, out_dim):
        stdv = 1.0 / math.sqrt(input.shape[1] * 1.0)
        out = fluid.layers.fc(input=input,
                              size=out_dim,
                              act='softmax',
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv)))
        return out
