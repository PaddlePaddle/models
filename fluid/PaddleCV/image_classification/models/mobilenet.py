from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

__all__ = ['MobileNet']

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class MobileNet():
    def __init__(self):
        self.params = train_parameters

    def net(self, input, class_dim=1000, scale=1.0):
        # conv1: 112x112
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        # 56x56
        input = self.depthwise_separable(
            input,
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale)

        input = self.depthwise_separable(
            input,
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=2,
            scale=scale)

        # 28x28
        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale)

        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=2,
            scale=scale)

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale)

        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=2,
            scale=scale)

        # 14x14
        for i in range(5):
            input = self.depthwise_separable(
                input,
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                scale=scale)
        # 7x7
        input = self.depthwise_separable(
            input,
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=2,
            scale=scale)

        input = self.depthwise_separable(
            input,
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=1,
            scale=scale)

        input = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_stride=1,
            pool_type='avg',
            global_pooling=True)

        output = fluid.layers.fc(input=input,
                                 size=class_dim,
                                 act='softmax',
                                 param_attr=ParamAttr(initializer=MSRA()))
        return output

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      channels=None,
                      num_groups=1,
                      act='relu',
                      use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(initializer=MSRA()),
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def depthwise_separable(self, input, num_filters1, num_filters2, num_groups,
                            stride, scale):
        depthwise_conv = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False)

        pointwise_conv = self.conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)
        return pointwise_conv
