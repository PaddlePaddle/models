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
from paddle.fluid.param_attr import ParamAttr

__all__ = ['GoogLeNet']


class GoogLeNet():
    def __init__(self):

        pass

    def conv_layer(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   groups=1,
                   act=None,
                   name=None):
        channels = input.shape[1]
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=False,
            name=name)
        return conv

    def xavier(self, channels, filter_size, name):
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv),
            name=name + "_weights")

        return param_attr

    def inception(self,
                  input,
                  channels,
                  filter1,
                  filter3R,
                  filter3,
                  filter5R,
                  filter5,
                  proj,
                  name=None):
        conv1 = self.conv_layer(
            input=input,
            num_filters=filter1,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_1x1")
        conv3r = self.conv_layer(
            input=input,
            num_filters=filter3R,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_3x3_reduce")
        conv3 = self.conv_layer(
            input=conv3r,
            num_filters=filter3,
            filter_size=3,
            stride=1,
            act=None,
            name="inception_" + name + "_3x3")
        conv5r = self.conv_layer(
            input=input,
            num_filters=filter5R,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_5x5_reduce")
        conv5 = self.conv_layer(
            input=conv5r,
            num_filters=filter5,
            filter_size=5,
            stride=1,
            act=None,
            name="inception_" + name + "_5x5")
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=1,
            pool_padding=1,
            pool_type='max')
        convprj = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=proj,
            stride=1,
            padding=0,
            name="inception_" + name + "_3x3_proj",
            param_attr=ParamAttr(
                name="inception_" + name + "_3x3_proj_weights"),
            bias_attr=False)
        cat = fluid.layers.concat(input=[conv1, conv3, conv5, convprj], axis=1)
        cat = fluid.layers.relu(cat)
        return cat

    def net(self, input, class_dim=1000):
        conv = self.conv_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act=None,
            name="conv1")
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        conv = self.conv_layer(
            input=pool,
            num_filters=64,
            filter_size=1,
            stride=1,
            act=None,
            name="conv2_1x1")
        conv = self.conv_layer(
            input=conv,
            num_filters=192,
            filter_size=3,
            stride=1,
            act=None,
            name="conv2_3x3")
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        ince3a = self.inception(pool, 192, 64, 96, 128, 16, 32, 32, "ince3a")
        ince3b = self.inception(ince3a, 256, 128, 128, 192, 32, 96, 64,
                                "ince3b")
        pool3 = fluid.layers.pool2d(
            input=ince3b, pool_size=3, pool_type='max', pool_stride=2)

        ince4a = self.inception(pool3, 480, 192, 96, 208, 16, 48, 64, "ince4a")
        ince4b = self.inception(ince4a, 512, 160, 112, 224, 24, 64, 64,
                                "ince4b")
        ince4c = self.inception(ince4b, 512, 128, 128, 256, 24, 64, 64,
                                "ince4c")
        ince4d = self.inception(ince4c, 512, 112, 144, 288, 32, 64, 64,
                                "ince4d")
        ince4e = self.inception(ince4d, 528, 256, 160, 320, 32, 128, 128,
                                "ince4e")
        pool4 = fluid.layers.pool2d(
            input=ince4e, pool_size=3, pool_type='max', pool_stride=2)

        ince5a = self.inception(pool4, 832, 256, 160, 320, 32, 128, 128,
                                "ince5a")
        ince5b = self.inception(ince5a, 832, 384, 192, 384, 48, 128, 128,
                                "ince5b")
        pool5 = fluid.layers.pool2d(
            input=ince5b, pool_size=7, pool_type='avg', pool_stride=7)
        dropout = fluid.layers.dropout(x=pool5, dropout_prob=0.4)
        out = fluid.layers.fc(input=dropout,
                              size=class_dim,
                              act='softmax',
                              param_attr=self.xavier(1024, 1, "out"),
                              name="out",
                              bias_attr=ParamAttr(name="out_offset"))

        pool_o1 = fluid.layers.pool2d(
            input=ince4a, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o1 = self.conv_layer(
            input=pool_o1,
            num_filters=128,
            filter_size=1,
            stride=1,
            act=None,
            name="conv_o1")
        fc_o1 = fluid.layers.fc(input=conv_o1,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1, "fc_o1"),
                                name="fc_o1",
                                bias_attr=ParamAttr(name="fc_o1_offset"))
        dropout_o1 = fluid.layers.dropout(x=fc_o1, dropout_prob=0.7)
        out1 = fluid.layers.fc(input=dropout_o1,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1, "out1"),
                               name="out1",
                               bias_attr=ParamAttr(name="out1_offset"))

        pool_o2 = fluid.layers.pool2d(
            input=ince4d, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o2 = self.conv_layer(
            input=pool_o2,
            num_filters=128,
            filter_size=1,
            stride=1,
            act=None,
            name="conv_o2")
        fc_o2 = fluid.layers.fc(input=conv_o2,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1, "fc_o2"),
                                name="fc_o2",
                                bias_attr=ParamAttr(name="fc_o2_offset"))
        dropout_o2 = fluid.layers.dropout(x=fc_o2, dropout_prob=0.7)
        out2 = fluid.layers.fc(input=dropout_o2,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1, "out2"),
                               name="out2",
                               bias_attr=ParamAttr(name="out2_offset"))

        # last fc layer is "out"
        return out, out1, out2
