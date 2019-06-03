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

import math
import six

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable
from paddle.fluid.regularizer import L2Decay

from ..registry import Backbones
from .base import BackboneBase

__all__ = ['DarkNet53Backbone']


class DarkNet(object):
    def __init__(self, depth, bn_decay=True):
        """
        Args:
            depth (int): DarkNet depth, should be 53.
            bn_decay (bool): whether perform L2Decay in batch_norm
        """
        if depth not in [53]:
            raise ValueError("depth {} not in [53].".format(depth))
        self.depth = depth
        self.bn_decay = bn_decay
        self.depth_cfg = {
                53: ([1,2,8,8,4], self.basicblock)
        }


    def _conv_norm(self,
                   input,
                   ch_out,
                   filter_size,
                   stride,
                   padding,
                   act='leaky',
                   name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name+".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        bn_param_attr = ParamAttr(regularizer=L2Decay(float(self.bn_decay)),
                                  name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(regularizer=L2Decay(float(self.bn_decay)),
                                 name=bn_name + '.offset')

        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        # leaky relu here has `alpha` as 0.1, can not be set by 
        # `act` param in fluid.layers.batch_norm above.
        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)

        return out

    def _downsample(self,
                    input, 
                    ch_out, 
                    filter_size=3, 
                    stride=2, 
                    padding=1, 
                    name=None):
        return self._conv_norm(input, 
                               ch_out=ch_out, 
                               filter_size=filter_size, 
                               stride=stride, 
                               padding=padding, 
                               name=name)

    def basicblock(self, input, ch_out, name=None):
        conv1 = self._conv_norm(input, 
                                ch_out=ch_out, 
                                filter_size=1, 
                                stride=1, 
                                padding=0, 
                                name=name+".0")
        conv2 = self._conv_norm(conv1, 
                                ch_out=ch_out*2, 
                                filter_size=3, 
                                stride=1, 
                                padding=1, 
                                name=name+".1")
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        return out

    def layer_warp(self, block_func, input, ch_out, count, name=None):
        out = block_func(input, 
                         ch_out=ch_out, 
                         name='{}.0'.format(name))
        for j in six.moves.xrange(1, count):
            out = block_func(out, 
                             ch_out=ch_out, 
                             name='{}.{}'.format(name, j))
        return out

    def get_backbone(self, body_input):
        """
        Get the backbone of DarkNet. We define DarkNet has 5 stages output.

        Args:
            body_input (Variable): input variable.

        Returns:
            The last variables of each stage.
        """
        stages, block_func = self.depth_cfg[self.depth]
        stages = stages[0:5]
        conv = self._conv_norm(input=body_input, 
                               ch_out=32, 
                               filter_size=3, 
                               stride=1, 
                               padding=1, 
                               name="yolo_input")
        downsample_ = self._downsample(input=conv, 
                                       ch_out=conv.shape[1]*2, 
                                       name="yolo_input.downsample")
        blocks = []
        for i, stage in enumerate(stages):
            block = self.layer_warp(block_func=block_func, 
                                    input=downsample_, 
                                    ch_out=32 *(2**i), 
                                    count=stage, 
                                    name="stage.{}".format(i))
            blocks.append(block)
            if i < len(stages) - 1: # do not downsaple in the last stage
                downsample_ = self._downsample(input=block, 
                                               ch_out=block.shape[1]*2, 
                                               name="stage.{}.downsample".format(i))
        return blocks



@Backbones.register
class DarkNet53Backbone(BackboneBase):
    def __init__(self, cfg):
        """
        Get the DarkNet53 backbone. We define DarkNet53 has 5 stages,
        from 1 to 5.

        Args:
            cfg (AttrDict): the config from given config filename.
        """
        super(DarkNet53Backbone, self).__init__(cfg)
        self.bn_decay = getattr(cfg.OPTIMIZER.WEIGHT_DECAY, 
                                'BN_DECAY', True)
        self.depth = 53

    def __call__(self, input):
        """
        Args:
            input (Variable): input variable.

        Returns:
            The last variables of each stage.
        """
        if not isinstance(input, Variable):
            raise TypeError(str(input) + " should be Variable")

        model = DarkNet(depth=self.depth, 
                        bn_decay=self.bn_decay)
        return model.get_backbone(input)

