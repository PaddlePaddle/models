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
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay

from ..registry import Necks

__all__ = ['FPN']


@Necks.register
class FPN(object):
    """
    FPN class
    Args:
        cfg (Dict): All parameters in dictionary.
    
    TODO(guanzhong): add more comments here.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.fpn_dim = cfg.FPN.DIM
        self.spatial_scale = list(map(eval, cfg.FPN.SPATIAL_SCALE))
        self.fpn_inner_output = None

    def _add_topdown_lateral(self, body_name, body_input, upper_output):
        lateral_name = 'fpn_inner_' + body_name + '_lateral'
        topdown_name = 'fpn_topdown_' + body_name
        fpn_inner_name = 'fpn_inner_' + body_name
        lateral = fluid.layers.conv2d(
            body_input,
            self.fpn_dim,
            1,
            param_attr=ParamAttr(
                name=lateral_name + "_w", initializer=Xavier()),
            bias_attr=ParamAttr(
                name=lateral_name + "_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)),
            name=lateral_name)
        shape = fluid.layers.shape(upper_output)
        shape_hw = fluid.layers.slice(shape, axes=[0], starts=[2], ends=[4])
        out_shape_ = shape_hw * 2
        out_shape = fluid.layers.cast(out_shape_, dtype='int32')
        out_shape.stop_gradient = True
        topdown = fluid.layers.resize_nearest(
            upper_output, scale=2., actual_shape=out_shape, name=topdown_name)

        return lateral + topdown

    def get_output(self, body_dict, body_name_list):
        """
        Add FPN neck onto backbone.

        Args:
            body_dict(Dict): Dictionary of variables and each element is the 
                output of backbone.
            body_name_list(List): List of names and each element represents 
                the name of each output of backbone.

        Return:
            fpn_dict(Dict): A dictionary represents the output of FPN neck with 
                their name.
            spatial_scale(List): A list of multiplicative spatial scale factor.
            fpn_name_list(List): A list of names regarding to output of FPN neck.
        """
        body_name_list = body_name_list[::-1]
        max_level = max(self.cfg.FPN.RPN_MAX_LEVEL,
                        self.cfg.ROI_EXTRACTOR.FPN_ROI_MAX_LEVEL)
        min_level = min(self.cfg.FPN.RPN_MIN_LEVEL,
                        self.cfg.ROI_EXTRACTOR.FPN_ROI_MIN_LEVEL)
        num_backbone_stages = len(body_name_list) - (
            min_level - self.cfg.MODEL.LOWEST_BACKBONE_LVL)
        body_name_list = body_name_list[:num_backbone_stages]
        self.spatial_scale = self.spatial_scale[:num_backbone_stages]
        self.fpn_inner_output = [[] for _ in range(num_backbone_stages)]
        fpn_inner_name = 'fpn_inner_' + body_name_list[0]
        self.fpn_inner_output[0] = fluid.layers.conv2d(
            body_dict[body_name_list[0]],
            self.fpn_dim,
            1,
            param_attr=ParamAttr(
                name=fpn_inner_name + "_w", initializer=Xavier()),
            bias_attr=ParamAttr(
                name=fpn_inner_name + "_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)),
            name=fpn_inner_name)
        for i in range(1, num_backbone_stages):
            body_name = body_name_list[i]
            body_input = body_dict[body_name]
            top_output = self.fpn_inner_output[i - 1]
            fpn_inner_single = self._add_topdown_lateral(body_name, body_input,
                                                         top_output)
            self.fpn_inner_output[i] = fpn_inner_single
        fpn_dict = {}
        fpn_name_list = []
        for i in range(num_backbone_stages):
            fpn_name = 'fpn_' + body_name_list[i]
            fpn_output = fluid.layers.conv2d(
                self.fpn_inner_output[i],
                self.fpn_dim,
                filter_size=3,
                padding=1,
                param_attr=ParamAttr(
                    name=fpn_name + "_w", initializer=Xavier()),
                bias_attr=ParamAttr(
                    name=fpn_name + "_b",
                    learning_rate=2.,
                    regularizer=L2Decay(0.)),
                name=fpn_name)
            fpn_dict[fpn_name] = fpn_output
            fpn_name_list.append(fpn_name)
        if max_level == self.cfg.MODEL.HIGHEST_BACKBONE_LVL + 1:
            body_top_name = fpn_name_list[0]
            body_top_extension = fluid.layers.pool2d(
                fpn_dict[body_top_name],
                1,
                'max',
                pool_stride=2,
                name=body_top_name + '_subsampled_2x')
            fpn_dict[body_top_name + '_subsampled_2x'] = body_top_extension
            fpn_name_list.insert(0, body_top_name + '_subsampled_2x')
            self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)
        return fpn_dict, self.spatial_scale, fpn_name_list
