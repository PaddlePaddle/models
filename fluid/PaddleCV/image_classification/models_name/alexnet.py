from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import math

__all__ = ['AlexNet']

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [40, 70, 100],
        "steps": [0.01, 0.001, 0.0001, 0.00001]
    }
}


class AlexNet():
    def __init__(self):
        self.params = train_parameters

    def net(self, input, class_dim=1000):
        stdv = 1.0 / math.sqrt(input.shape[1] * 11 * 11)
        layer_name = [
            "conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "fc8"
        ]
        conv1 = fluid.layers.conv2d(
            input=input,
            num_filters=64,
            filter_size=11,
            stride=4,
            padding=2,
            groups=1,
            act='relu',
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[0] + "_offset"),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[0] + "_weights"))
        pool1 = fluid.layers.pool2d(
            input=conv1,
            pool_size=3,
            pool_stride=2,
            pool_padding=0,
            pool_type='max')

        stdv = 1.0 / math.sqrt(pool1.shape[1] * 5 * 5)
        conv2 = fluid.layers.conv2d(
            input=pool1,
            num_filters=192,
            filter_size=5,
            stride=1,
            padding=2,
            groups=1,
            act='relu',
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[1] + "_offset"),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[1] + "_weights"))
        pool2 = fluid.layers.pool2d(
            input=conv2,
            pool_size=3,
            pool_stride=2,
            pool_padding=0,
            pool_type='max')

        stdv = 1.0 / math.sqrt(pool2.shape[1] * 3 * 3)
        conv3 = fluid.layers.conv2d(
            input=pool2,
            num_filters=384,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act='relu',
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[2] + "_offset"),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[2] + "_weights"))

        stdv = 1.0 / math.sqrt(conv3.shape[1] * 3 * 3)
        conv4 = fluid.layers.conv2d(
            input=conv3,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act='relu',
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[3] + "_offset"),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[3] + "_weights"))

        stdv = 1.0 / math.sqrt(conv4.shape[1] * 3 * 3)
        conv5 = fluid.layers.conv2d(
            input=conv4,
            num_filters=256,
            filter_size=3,
            stride=1,
            padding=1,
            groups=1,
            act='relu',
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[4] + "_offset"),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[4] + "_weights"))
        pool5 = fluid.layers.pool2d(
            input=conv5,
            pool_size=3,
            pool_stride=2,
            pool_padding=0,
            pool_type='max')

        drop6 = fluid.layers.dropout(x=pool5, dropout_prob=0.5)
        stdv = 1.0 / math.sqrt(drop6.shape[1] * drop6.shape[2] *
                               drop6.shape[3] * 1.0)

        fc6 = fluid.layers.fc(
            input=drop6,
            size=4096,
            act='relu',
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[5] + "_offset"),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[5] + "_weights"))

        drop7 = fluid.layers.dropout(x=fc6, dropout_prob=0.5)
        stdv = 1.0 / math.sqrt(drop7.shape[1] * 1.0)

        fc7 = fluid.layers.fc(
            input=drop7,
            size=4096,
            act='relu',
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[6] + "_offset"),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[6] + "_weights"))

        stdv = 1.0 / math.sqrt(fc7.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=fc7,
            size=class_dim,
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[7] + "_offset"),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=layer_name[7] + "_weights"))
        return out
