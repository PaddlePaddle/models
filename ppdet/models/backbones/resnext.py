# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from ppdet.core.workspace import register, serializable
from .resnet import ResNet

__all__ = ['ResNeXt']


@register
@serializable
class ResNeXt(ResNet):
    """
    ResNeXt, see https://arxiv.org/abs/1611.05431
    Args:
        depth (int): network depth, should be 50, 101, 152.
        groups (int): group convolution cardinality
        group_width (int): width of each group convolution
        freeze_at (int): freeze the backbone at which stage
        freeze_bn (bool): fix batch norm weights
        affine_channel (bool): use batch_norm or affine_channel.
        bn_decay (bool): apply weight decay to in batch norm weights
        variant (str): ResNet variant, supports 'a', 'b', 'c', 'd' currently
        feature_maps (list): index of the stages whose feature maps are returned
    """

    def __init__(self,
                 depth=50,
                 groups=64,
                 group_width=4,
                 freeze_at=2,
                 freeze_bn=True,
                 affine_channel=True,
                 bn_decay=True,
                 variant='a',
                 feature_maps=[2, 3, 4, 5]):
        assert depth in [50, 101, 152], "depth {} should be 50, 101 or 152"
        super(ResNeXt, self).__init__(depth, freeze_at, freeze_bn,
                                      affine_channel, bn_decay, variant,
                                      feature_maps)
        self.depth_cfg = {
            50: ([3, 4, 6, 3], self.bottleneck),
            101: ([3, 4, 23, 3], self.bottleneck),
            152: ([3, 8, 36, 3], self.bottleneck)
        }
        self.stage_filters = [256, 512, 1024, 2048]
        self.groups = groups
        self.group_width = group_width
        self._model_type = 'ResNeXt'


@register
@serializable
class ResNeXtC5(ResNeXt):
    __doc__ = ResNeXt.__doc__

    def __init__(self,
                 depth=50,
                 groups=64,
                 group_width=4,
                 freeze_at=2,
                 freeze_bn=True,
                 affine_channel=True,
                 bn_decay=True,
                 variant='b',
                 feature_maps=[5]):
        super(ResNeXtC5, self).__init__(depth, groups, group_width, freeze_at,
                                        freeze_bn, affine_channel, bn_decay,
                                        variant, feature_maps)
        self.severed_head = True
