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
from paddle.fluid.framework import Variable

from ..registry import Backbones
from ..registry import BBoxHeadConvs
from .base import BackboneBase
from .resnet import ResNet

__all__ = ['ResNeXt101Backbone', 'ResNeXt101C5']


class ResNeXt(ResNet):
    def __init__(self, depth, freeze_bn, affine_channel, groups):
        """
        Args:
            depth (int): ResNet depth, should be 50, 101, 152.
            freeze_bn (bool): whether to fix batch norm
                (meaning the scale and bias does not update).
            affine_channel (bool): Use batch_norm or affine_channel.
            groups (int): define group parammeter for bottleneck
        """
        if depth not in [50, 101, 152]:
            raise ValueError("depth {} not in [50, 101, 152].".format(depth))

        self.depth = depth
        self.freeze_bn = freeze_bn
        self.affine_channel = affine_channel
        self.depth_cfg = {
            50: ([3, 4, 6, 3], self.bottleneck),
            101: ([3, 4, 23, 3], self.bottleneck),
            152: ([3, 8, 36, 3], self.bottleneck)
        }
        self.stage_filters = [256, 512, 1024, 2048]
        self.groups = groups

    def bottleneck(self, input, num_filters, stride, is_first, name):
        conv0 = self._conv_norm(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        conv1 = self._conv_norm(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=self.groups,
            act='relu',
            name=name + "_branch2b")
        conv2 = self._conv_norm(
            input=conv1,
            num_filters=num_filters,
            filter_size=1,
            act=None,
            name=name + "_branch2c")
        short = self._shortcut(
            input,
            num_filters,
            stride,
            is_first=is_first,
            name=name + "_branch1")
        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")


@Backbones.register
class ResNeXt101Backbone(BackboneBase):
    def __init__(self, cfg):
        super(ResNeXt101Backbone, self).__init__(cfg)
        self.freeze_at = getattr(cfg.MODEL, 'FREEZE_AT', 2)
        assert self.freeze_at in [0, 1, 2, 3, 4
                                  ], "The freeze_at should be 0, 1, 2, 3, or 4"
        self.freeze_bn = getattr(cfg.MODEL, 'FREEZE_BN', False)
        self.affine_channel = getattr(cfg.MODEL, 'AFFINE_CHANNEL', False)
        self.groups = getattr(cfg.MODEL, "GROUPS", 64)
        self.endpoint = getattr(cfg.MODEL, 'ENDPOINT', 4)
        self.number = 101

    def __call__(self, input):
        if not isinstance(input, Variable):
            raise TypeError(str(input) + " shouble be Variable")

        model = ResNeXt(self.number, self.freeze_bn, self.affine_channel,
                        self.groups)
        return model.get_backbone(input, self.endpoint, self.freeze_at)


@BBoxHeadConvs.register
class ResNeXt101C5(object):
    def __init__(self, cfg):
        self.freeze_bn = getattr(cfg.MODEL, 'FREEZE_BN', False)
        self.affine_channel = getattr(cfg.MODEL, 'AFFINE_CHANNEL', False)
        self.groups = getattr(cfg.MODEL, 'GROUPS', 64)
        self.number = 101
        self.stage_number = 5

    def __call__(self, input):
        model = ResNeXt(self.number, self.freeze_bn, self.affine_channel,
                        self.groups)
        res5 = model.layer_warp(input, self.stage_number)
        feat = fluid.layers.pool2d(
            input=res5, pool_type='avg', pool_size=7, name='res5_pool')
        return feat
