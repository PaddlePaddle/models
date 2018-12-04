from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "SE_ResNeXt", "SE_ResNeXt50_32x4d", "SE_ResNeXt101_32x4d",
    "SE_ResNeXt152_32x4d"
]

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "dropout_seed": None,
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [40, 80, 100],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class SE_ResNeXt():
    def __init__(self, layers=50):
        self.params = train_parameters
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name='conv1', )
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name="conv1", )
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
                name='conv1')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv2')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv3')
            conv = fluid.layers.pool2d(
                input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
                pool_type='max')
        n = 1 if layers == 50 or layers == 101 else 3
        for block in range(len(depth)):
            n += 1
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    reduction_ratio=reduction_ratio,
                    name=str(n) + '_' + str(i + 1))

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)
        drop = fluid.layers.dropout(
            x=pool, dropout_prob=0.5, seed=self.params['dropout_seed'])
        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=drop,
            size=class_dim,
            act='softmax',
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name='fc6_weights'),
            bias_attr=ParamAttr(name='fc6_offset'))
        return out

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            filter_size = 1
            return self.conv_bn_layer(
                input, ch_out, filter_size, stride, name='conv' + name + '_prj')
        else:
            return input

    def bottleneck_block(self,
                         input,
                         num_filters,
                         stride,
                         cardinality,
                         reduction_ratio,
                         name=None):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name='conv' + name + '_x1')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu',
            name='conv' + name + '_x2')
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            name='conv' + name + '_x3')
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio,
            name='fc' + name)

        short = self.shortcut(input, num_filters * 2, stride, name=name)

        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=ParamAttr(name=name + '_weights'), )
        bn_name = name + "_bn"
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale


def SE_ResNeXt50_32x4d():
    model = SE_ResNeXt(layers=50)
    return model


def SE_ResNeXt101_32x4d():
    model = SE_ResNeXt(layers=101)
    return model


def SE_ResNeXt152_32x4d():
    model = SE_ResNeXt(layers=152)
    return model
