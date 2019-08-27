#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import math

__all__ = ['InceptionV4']

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


class InceptionV4():
    def __init__(self):
        self.params = train_parameters

    def net(self, input, class_dim=1000):
        x = self.inception_stem(input)

        for i in range(4):
            x = self.inceptionA(x)
        x = self.reductionA(x)

        for i in range(7):
            x = self.inceptionB(x)
        x = self.reductionB(x)

        for i in range(3):
            x = self.inceptionC(x)

        pool = fluid.layers.pool2d(
            input=x, pool_size=8, pool_type='avg', global_pooling=True)

        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)

        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=drop,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))
        return out

    def conv_bn_layer(self,
                      data,
                      num_filters,
                      filter_size,
                      stride=1,
                      padding=0,
                      groups=1,
                      act='relu'):
        conv = fluid.layers.conv2d(
            input=data,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def inception_stem(self, data):
        conv = self.conv_bn_layer(data, 32, 3, stride=2, act='relu')
        conv = self.conv_bn_layer(conv, 32, 3, act='relu')
        conv = self.conv_bn_layer(conv, 64, 3, padding=1, act='relu')

        pool1 = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_stride=2, pool_type='max')
        conv2 = self.conv_bn_layer(conv, 96, 3, stride=2, act='relu')
        concat = fluid.layers.concat([pool1, conv2], axis=1)

        conv1 = self.conv_bn_layer(concat, 64, 1, act='relu')
        conv1 = self.conv_bn_layer(conv1, 96, 3, act='relu')

        conv2 = self.conv_bn_layer(concat, 64, 1, act='relu')
        conv2 = self.conv_bn_layer(
            conv2, 64, (7, 1), padding=(3, 0), act='relu')
        conv2 = self.conv_bn_layer(
            conv2, 64, (1, 7), padding=(0, 3), act='relu')
        conv2 = self.conv_bn_layer(conv2, 96, 3, act='relu')

        concat = fluid.layers.concat([conv1, conv2], axis=1)

        conv1 = self.conv_bn_layer(concat, 192, 3, stride=2, act='relu')
        pool1 = fluid.layers.pool2d(
            input=concat, pool_size=3, pool_stride=2, pool_type='max')

        concat = fluid.layers.concat([conv1, pool1], axis=1)

        return concat

    def inceptionA(self, data):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_padding=1, pool_type='avg')
        conv1 = self.conv_bn_layer(pool1, 96, 1, act='relu')

        conv2 = self.conv_bn_layer(data, 96, 1, act='relu')

        conv3 = self.conv_bn_layer(data, 64, 1, act='relu')
        conv3 = self.conv_bn_layer(conv3, 96, 3, padding=1, act='relu')

        conv4 = self.conv_bn_layer(data, 64, 1, act='relu')
        conv4 = self.conv_bn_layer(conv4, 96, 3, padding=1, act='relu')
        conv4 = self.conv_bn_layer(conv4, 96, 3, padding=1, act='relu')

        concat = fluid.layers.concat([conv1, conv2, conv3, conv4], axis=1)

        return concat

    def reductionA(self, data):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_stride=2, pool_type='max')

        conv2 = self.conv_bn_layer(data, 384, 3, stride=2, act='relu')

        conv3 = self.conv_bn_layer(data, 192, 1, act='relu')
        conv3 = self.conv_bn_layer(conv3, 224, 3, padding=1, act='relu')
        conv3 = self.conv_bn_layer(conv3, 256, 3, stride=2, act='relu')

        concat = fluid.layers.concat([pool1, conv2, conv3], axis=1)

        return concat

    def inceptionB(self, data):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_padding=1, pool_type='avg')
        conv1 = self.conv_bn_layer(pool1, 128, 1, act='relu')

        conv2 = self.conv_bn_layer(data, 384, 1, act='relu')

        conv3 = self.conv_bn_layer(data, 192, 1, act='relu')
        conv3 = self.conv_bn_layer(
            conv3, 224, (1, 7), padding=(0, 3), act='relu')
        conv3 = self.conv_bn_layer(
            conv3, 256, (7, 1), padding=(3, 0), act='relu')

        conv4 = self.conv_bn_layer(data, 192, 1, act='relu')
        conv4 = self.conv_bn_layer(
            conv4, 192, (1, 7), padding=(0, 3), act='relu')
        conv4 = self.conv_bn_layer(
            conv4, 224, (7, 1), padding=(3, 0), act='relu')
        conv4 = self.conv_bn_layer(
            conv4, 224, (1, 7), padding=(0, 3), act='relu')
        conv4 = self.conv_bn_layer(
            conv4, 256, (7, 1), padding=(3, 0), act='relu')

        concat = fluid.layers.concat([conv1, conv2, conv3, conv4], axis=1)

        return concat

    def reductionB(self, data):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_stride=2, pool_type='max')

        conv2 = self.conv_bn_layer(data, 192, 1, act='relu')
        conv2 = self.conv_bn_layer(conv2, 192, 3, stride=2, act='relu')

        conv3 = self.conv_bn_layer(data, 256, 1, act='relu')
        conv3 = self.conv_bn_layer(
            conv3, 256, (1, 7), padding=(0, 3), act='relu')
        conv3 = self.conv_bn_layer(
            conv3, 320, (7, 1), padding=(3, 0), act='relu')
        conv3 = self.conv_bn_layer(conv3, 320, 3, stride=2, act='relu')

        concat = fluid.layers.concat([pool1, conv2, conv3], axis=1)

        return concat

    def inceptionC(self, data):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_padding=1, pool_type='avg')
        conv1 = self.conv_bn_layer(pool1, 256, 1, act='relu')

        conv2 = self.conv_bn_layer(data, 256, 1, act='relu')

        conv3 = self.conv_bn_layer(data, 384, 1, act='relu')
        conv3_1 = self.conv_bn_layer(
            conv3, 256, (1, 3), padding=(0, 1), act='relu')
        conv3_2 = self.conv_bn_layer(
            conv3, 256, (3, 1), padding=(1, 0), act='relu')

        conv4 = self.conv_bn_layer(data, 384, 1, act='relu')
        conv4 = self.conv_bn_layer(
            conv4, 448, (1, 3), padding=(0, 1), act='relu')
        conv4 = self.conv_bn_layer(
            conv4, 512, (3, 1), padding=(1, 0), act='relu')
        conv4_1 = self.conv_bn_layer(
            conv4, 256, (1, 3), padding=(0, 1), act='relu')
        conv4_2 = self.conv_bn_layer(
            conv4, 256, (3, 1), padding=(1, 0), act='relu')

        concat = fluid.layers.concat(
            [conv1, conv2, conv3_1, conv3_2, conv4_1, conv4_2], axis=1)

        return concat
