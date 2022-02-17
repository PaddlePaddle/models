from paddle import fluid
from paddle.fluid import dygraph
from paddle.fluid.dygraph import nn

from pytracking.libs.Fconv2d import FConv2D


class SiamFCEstimator(dygraph.layers.Layer):
    def __init__(self, name):
        super().__init__(name)
        init_w = fluid.ParamAttr(
            name="a_weight",
            initializer=fluid.initializer.ConstantInitializer(0.001),
            learning_rate=0.,
            trainable=False)
        init_b = fluid.ParamAttr(
            name="a_bias",
            initializer=fluid.initializer.ConstantInitializer(0.),
            trainable=True)

        self.adjust_conv = nn.Conv2D(
            1, 1, 1, 1, 0, param_attr=init_w, bias_attr=init_b)

    def forward(self, exemplar, instance):
        exemplar_f = self.get_reference(exemplar)
        instance_f = self.get_search_feat(instance)
        score_map = self.estimate(exemplar_f, instance_f)
        return score_map

    def get_reference(self, feat):
        # remove list warp
        return feat[0]

    def get_search_feat(self, feat):
        # remove list warp
        return feat[0]

    def estimate(self, exemplar, instance):
        shape = instance.shape
        instance = fluid.layers.reshape(
            instance, shape=[1, -1, shape[2], shape[3]])

        score_map = FConv2D(instance, exemplar, stride=1, padding=0, dilation=1, groups=shape[0])
        score_map = fluid.layers.transpose(score_map, [1, 0, 2, 3])
        score_map = self.adjust_conv(score_map)
        return score_map
