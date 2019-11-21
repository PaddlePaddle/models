#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "DenseNet", "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
    "DenseNet264"
]


class DenseNet():
    def __init__(self, layers=121):
        self.layers = layers

    def net(self, input, bn_size=4, dropout=0, class_dim=1000):
        layers = self.layers
        supported_layers = [121, 161, 169, 201, 264]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        densenet_spec = {
            121: (64, 32, [6, 12, 24, 16]),
            161: (96, 48, [6, 12, 36, 24]),
            169: (64, 32, [6, 12, 32, 32]),
            201: (64, 32, [6, 12, 48, 32]),
            264: (64, 32, [6, 12, 64, 48])
        }

        num_init_features, growth_rate, block_config = densenet_spec[layers]
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_init_features,
            filter_size=7,
            stride=2,
            padding=3,
            act=None,
            param_attr=ParamAttr(name="conv1_weights"),
            bias_attr=False)
        conv = fluid.layers.batch_norm(
            input=conv,
            act='relu',
            param_attr=ParamAttr(name='conv1_bn_scale'),
            bias_attr=ParamAttr(name='conv1_bn_offset'),
            moving_mean_name='conv1_bn_mean',
            moving_variance_name='conv1_bn_variance')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            conv = self.make_dense_block(
                conv,
                num_layers,
                bn_size,
                growth_rate,
                dropout,
                name='conv' + str(i + 2))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                conv = self.make_transition(
                    conv, num_features // 2, name='conv' + str(i + 2) + '_blk')
                num_features = num_features // 2
        conv = fluid.layers.batch_norm(
            input=conv,
            act='relu',
            param_attr=ParamAttr(name='conv5_blk_bn_scale'),
            bias_attr=ParamAttr(name='conv5_blk_bn_offset'),
            moving_mean_name='conv5_blk_bn_mean',
            moving_variance_name='conv5_blk_bn_variance')
        conv = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(conv.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=conv,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name="fc_weights"),
            bias_attr=ParamAttr(name='fc_offset'))
        return out

    def make_transition(self, input, num_output_features, name=None):
        bn_ac = fluid.layers.batch_norm(
            input,
            act='relu',
            param_attr=ParamAttr(name=name + '_bn_scale'),
            bias_attr=ParamAttr(name + '_bn_offset'),
            moving_mean_name=name + '_bn_mean',
            moving_variance_name=name + '_bn_variance')

        bn_ac_conv = fluid.layers.conv2d(
            input=bn_ac,
            num_filters=num_output_features,
            filter_size=1,
            stride=1,
            act=None,
            bias_attr=False,
            param_attr=ParamAttr(name=name + "_weights"))
        pool = fluid.layers.pool2d(
            input=bn_ac_conv, pool_size=2, pool_stride=2, pool_type='avg')
        return pool

    def make_dense_block(self,
                         input,
                         num_layers,
                         bn_size,
                         growth_rate,
                         dropout,
                         name=None):
        conv = input
        for layer in range(num_layers):
            conv = self.make_dense_layer(
                conv,
                growth_rate,
                bn_size,
                dropout,
                name=name + '_' + str(layer + 1))
        return conv

    def make_dense_layer(self, input, growth_rate, bn_size, dropout, name=None):
        bn_ac = fluid.layers.batch_norm(
            input,
            act='relu',
            param_attr=ParamAttr(name=name + '_x1_bn_scale'),
            bias_attr=ParamAttr(name + '_x1_bn_offset'),
            moving_mean_name=name + '_x1_bn_mean',
            moving_variance_name=name + '_x1_bn_variance')
        bn_ac_conv = fluid.layers.conv2d(
            input=bn_ac,
            num_filters=bn_size * growth_rate,
            filter_size=1,
            stride=1,
            act=None,
            bias_attr=False,
            param_attr=ParamAttr(name=name + "_x1_weights"))
        bn_ac = fluid.layers.batch_norm(
            bn_ac_conv,
            act='relu',
            param_attr=ParamAttr(name=name + '_x2_bn_scale'),
            bias_attr=ParamAttr(name + '_x2_bn_offset'),
            moving_mean_name=name + '_x2_bn_mean',
            moving_variance_name=name + '_x2_bn_variance')
        bn_ac_conv = fluid.layers.conv2d(
            input=bn_ac,
            num_filters=growth_rate,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            bias_attr=False,
            param_attr=ParamAttr(name=name + "_x2_weights"))
        if dropout:
            bn_ac_conv = fluid.layers.dropout(
                x=bn_ac_conv, dropout_prob=dropout)
        bn_ac_conv = fluid.layers.concat([input, bn_ac_conv], axis=1)
        return bn_ac_conv


def DenseNet121():
    model = DenseNet(layers=121)
    return model


def DenseNet161():
    model = DenseNet(layers=161)
    return model


def DenseNet169():
    model = DenseNet(layers=169)
    return model


def DenseNet201():
    model = DenseNet(layers=201)
    return model


def DenseNet264():
    model = DenseNet(layers=264)
    return model
