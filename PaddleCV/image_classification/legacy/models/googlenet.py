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

__all__ = ['GoogleNet']

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 70, 100],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class GoogleNet():
    def __init__(self):
        self.params = train_parameters

    def conv_layer(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   groups=1,
                   act=None):
        channels = input.shape[1]
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv))
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=False)
        return conv

    def xavier(self, channels, filter_size):
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = fluid.param_attr.ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv))
        return param_attr

    def inception(self, name, input, channels, filter1, filter3R, filter3,
                  filter5R, filter5, proj):
        conv1 = self.conv_layer(
            input=input, num_filters=filter1, filter_size=1, stride=1, act=None)
        conv3r = self.conv_layer(
            input=input,
            num_filters=filter3R,
            filter_size=1,
            stride=1,
            act=None)
        conv3 = self.conv_layer(
            input=conv3r,
            num_filters=filter3,
            filter_size=3,
            stride=1,
            act=None)
        conv5r = self.conv_layer(
            input=input,
            num_filters=filter5R,
            filter_size=1,
            stride=1,
            act=None)
        conv5 = self.conv_layer(
            input=conv5r,
            num_filters=filter5,
            filter_size=5,
            stride=1,
            act=None)
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=1,
            pool_padding=1,
            pool_type='max')
        convprj = fluid.layers.conv2d(
            input=pool, filter_size=1, num_filters=proj, stride=1, padding=0)
        cat = fluid.layers.concat(input=[conv1, conv3, conv5, convprj], axis=1)
        cat = fluid.layers.relu(cat)
        return cat

    def net(self, input, class_dim=1000):
        conv = self.conv_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act=None)
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        conv = self.conv_layer(
            input=pool, num_filters=64, filter_size=1, stride=1, act=None)
        conv = self.conv_layer(
            input=conv, num_filters=192, filter_size=3, stride=1, act=None)
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        ince3a = self.inception("ince3a", pool, 192, 64, 96, 128, 16, 32, 32)
        ince3b = self.inception("ince3b", ince3a, 256, 128, 128, 192, 32, 96,
                                64)
        pool3 = fluid.layers.pool2d(
            input=ince3b, pool_size=3, pool_type='max', pool_stride=2)

        ince4a = self.inception("ince4a", pool3, 480, 192, 96, 208, 16, 48, 64)
        ince4b = self.inception("ince4b", ince4a, 512, 160, 112, 224, 24, 64,
                                64)
        ince4c = self.inception("ince4c", ince4b, 512, 128, 128, 256, 24, 64,
                                64)
        ince4d = self.inception("ince4d", ince4c, 512, 112, 144, 288, 32, 64,
                                64)
        ince4e = self.inception("ince4e", ince4d, 528, 256, 160, 320, 32, 128,
                                128)
        pool4 = fluid.layers.pool2d(
            input=ince4e, pool_size=3, pool_type='max', pool_stride=2)

        ince5a = self.inception("ince5a", pool4, 832, 256, 160, 320, 32, 128,
                                128)
        ince5b = self.inception("ince5b", ince5a, 832, 384, 192, 384, 48, 128,
                                128)
        pool5 = fluid.layers.pool2d(
            input=ince5b, pool_size=7, pool_type='avg', pool_stride=7)
        dropout = fluid.layers.dropout(x=pool5, dropout_prob=0.4)
        out = fluid.layers.fc(input=dropout,
                              size=class_dim,
                              act='softmax',
                              param_attr=self.xavier(1024, 1))

        pool_o1 = fluid.layers.pool2d(
            input=ince4a, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o1 = self.conv_layer(
            input=pool_o1, num_filters=128, filter_size=1, stride=1, act=None)
        fc_o1 = fluid.layers.fc(input=conv_o1,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1))
        dropout_o1 = fluid.layers.dropout(x=fc_o1, dropout_prob=0.7)
        out1 = fluid.layers.fc(input=dropout_o1,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1))

        pool_o2 = fluid.layers.pool2d(
            input=ince4d, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o2 = self.conv_layer(
            input=pool_o2, num_filters=128, filter_size=1, stride=1, act=None)
        fc_o2 = fluid.layers.fc(input=conv_o2,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1))
        dropout_o2 = fluid.layers.dropout(x=fc_o2, dropout_prob=0.7)
        out2 = fluid.layers.fc(input=dropout_o2,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1))

        # last fc layer is "out"
        return out, out1, out2
