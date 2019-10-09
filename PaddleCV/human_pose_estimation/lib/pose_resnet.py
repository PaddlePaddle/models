# Copyright (c) 2018-present, Baidu, Inc.
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
##############################################################################

"""Functions for building network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid

__all__ = ["ResNet", "ResNet50", "ResNet101", "ResNet152"]

# Global parameters
BN_MOMENTUM = 0.9

class ResNet():
    def __init__(self, layers=50, kps_num=16, test_mode=False):
        """
        :param layers:  int, the layers number which is used here
        :param kps_num: int, the number of keypoints in accord with the dataset
        :param test_mode: bool, if True, only return output heatmaps, no loss

        :return: loss, output heatmaps
        """
        self.k = kps_num
        self.layers = layers
        self.test_mode = test_mode

    def net(self, input, target=None, target_weight=None):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1)

        conv = fluid.layers.conv2d_transpose(
                input=conv, num_filters=256,
                filter_size=4,
                padding=1,
                stride=2,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(0., 0.001)),
                act=None,
                bias_attr=False)
        conv = fluid.layers.batch_norm(input=conv, act='relu', momentum=BN_MOMENTUM)
        conv = fluid.layers.conv2d_transpose(
                input=conv, num_filters=256,
                filter_size=4,
                padding=1,
                stride=2,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(0., 0.001)),
                act=None,
                bias_attr=False)
        conv = fluid.layers.batch_norm(input=conv, act='relu', momentum=BN_MOMENTUM)
        conv = fluid.layers.conv2d_transpose(
                input=conv, num_filters=256,
                filter_size=4,
                padding=1,
                stride=2,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(0., 0.001)),
                act=None,
                bias_attr=False)
        conv = fluid.layers.batch_norm(input=conv, act='relu', momentum=BN_MOMENTUM)

        out = fluid.layers.conv2d(
                input=conv,
                num_filters=self.k,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Normal(0., 0.001)))

        if self.test_mode:
            return out
        else:
            loss = self.calc_loss(out, target, target_weight)
            return loss, out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.001)),
            act=None,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act, momentum=BN_MOMENTUM)

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def calc_loss(self, heatmap, target, target_weight):
        _, c, h, w = heatmap.shape
        x = fluid.layers.reshape(heatmap, (-1, self.k, h*w))
        y = fluid.layers.reshape(target,  (-1, self.k, h*w))
        w = fluid.layers.reshape(target_weight, (-1, self.k))

        x = fluid.layers.split(x, num_or_sections=self.k, dim=1)
        y = fluid.layers.split(y, num_or_sections=self.k, dim=1)
        w = fluid.layers.split(w, num_or_sections=self.k, dim=1)

        _list = []
        for idx in range(self.k):
            _tmp = fluid.layers.scale(x=x[idx] - y[idx], scale=1.)
            _tmp = _tmp * _tmp
            _tmp = fluid.layers.reduce_mean(_tmp, dim=2)
            _list.append(_tmp * w[idx])

        _loss = fluid.layers.concat(_list, axis=0)
        _loss = fluid.layers.reduce_mean(_loss)
        return 0.5 * _loss

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None)

        short = self.shortcut(input, num_filters * 4, stride)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')

def ResNet50():
    model = ResNet(layers=50)
    return model

def ResNet101():
    model = ResNet(layers=101)
    return model

def ResNet152():
    model = ResNet(layers=152)
    return model
