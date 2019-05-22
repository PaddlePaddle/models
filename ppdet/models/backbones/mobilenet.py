#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable

from ..registry import Backbones
from .base import BackboneBase

__all__ = ['MobileNetV1Backbone']


class MobileNet(object):
    def __init__(self):
        pass

    def _conv_norm(self,
                   input,
                   filter_size,
                   num_filters,
                   stride,
                   padding,
                   channels=None,
                   num_groups=1,
                   act='relu',
                   use_cudnn=True,
                   name=None):
        parameter_attr = ParamAttr(
            learning_rate=0.1,
            initializer=fluid.initializer.MSRA(),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        bn_name = name + "_bn"
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def depthwise_separable(self,
                            input,
                            num_filters1,
                            num_filters2,
                            num_groups,
                            stride,
                            scale,
                            name=None):
        depthwise_conv = self._conv_norm(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False,
            name=name + "_dw")

        pointwise_conv = self._conv_norm(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_seq")
        return pointwise_conv

    def get_backone(self, input, scale=1.0):
        """
        Args:
            scale (float): the scale of groups number/ filter number
        """
        # 300x300
        tmp = self._conv_norm(input, 3, int(32 * scale), 2, 1, 3, name="conv1")
        # 150x150
        tmp = self.depthwise_separable(tmp, 32, 64, 32, 1, scale, "conv2_1")
        tmp = self.depthwise_separable(tmp, 64, 128, 64, 2, scale, "conv2_2")
        # 75x75
        tmp = self.depthwise_separable(tmp, 128, 128, 128, 1, scale, "conv3_1")
        tmp = self.depthwise_separable(tmp, 128, 256, 128, 2, scale, "conv3_2")
        # 38x38
        tmp = self.depthwise_separable(tmp, 256, 256, 256, 1, scale, "conv4_1")
        tmp = self.depthwise_separable(tmp, 256, 512, 256, 2, scale, "conv4_2")
        # 19x19
        for i in range(5):
            tmp = self.depthwise_separable(tmp, 512, 512, 512, 1, scale,
                                           "conv5" + "_" + str(i + 1))
        module11 = tmp
        tmp = self.depthwise_separable(tmp, 512, 1024, 512, 2, scale, "conv5_6")
        # 10x10
        module13 = self.depthwise_separable(tmp, 1024, 1024, 1024, 1, scale,
                                            "conv6")
        return module11, module13


@Backbones.register
class MobileNetV1Backbone(BackboneBase):
    def __init__(self, cfg):
        super(MobileNetV1Backbone, self).__init__(cfg)
        self.scale = cfg.MODEL.CONV_GROUP_SCALE

    def __call__(self, input):
        """
        Get the backbone of MobileNetV1.
        Args:
            input (Variable): input variable.
        Returns:
            The two feature map of MobileNet.
        """
        model = MobileNet()
        return model.get_backone(input, self.scale)
