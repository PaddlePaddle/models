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

from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register

__all__ = ['FPN', 'ESFPN']


@register
class FPN(object):
    """
    Feature Pyramid Network, see https://arxiv.org/abs/1612.03144

    Args:
        num_chan (int): number of feature channels
        min_level (int): lowest level of the backbone feature map to use
        max_level (int): highest level of the backbone feature map to use
        spatial_scale (list): feature map scaling factor
        has_extra_convs (bool): whether has extral convolutions in higher levels
    """

    def __init__(self,
                 num_chan=256,
                 min_level=2,
                 max_level=6,
                 spatial_scale=[1. / 32., 1. / 16., 1. / 8., 1. / 4.],
                 has_extra_convs=False):
        self.num_chan = num_chan
        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale
        self.has_extra_convs = has_extra_convs

    def _add_topdown_lateral(self, body_name, body_input, upper_output):
        lateral_name = 'fpn_inner_' + body_name + '_lateral'
        topdown_name = 'fpn_topdown_' + body_name
        fan = body_input.shape[1]
        lateral = fluid.layers.conv2d(
            body_input,
            self.num_chan,
            1,
            param_attr=ParamAttr(
                name=lateral_name + "_w", initializer=Xavier(fan_out=fan)),
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

    def get_output(self, body_dict):
        """
        Add FPN onto backbone.

        Args:
            body_dict(OrderedDict): Dictionary of variables and each element is the
                output of backbone.

        Return:
            fpn_dict(OrderedDict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.
        """
        body_name_list = list(body_dict.keys())[::-1]
        num_backbone_stages = len(body_name_list)
        self.fpn_inner_output = [[] for _ in range(num_backbone_stages)]
        fpn_inner_name = 'fpn_inner_' + body_name_list[0]
        body_input = body_dict[body_name_list[0]]
        fan = body_input.shape[1]
        self.fpn_inner_output[0] = fluid.layers.conv2d(
            body_input,
            self.num_chan,
            1,
            param_attr=ParamAttr(
                name=fpn_inner_name + "_w", initializer=Xavier(fan_out=fan)),
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
            fan = self.fpn_inner_output[i].shape[1] * 3 * 3
            fpn_output = fluid.layers.conv2d(
                self.fpn_inner_output[i],
                self.num_chan,
                filter_size=3,
                padding=1,
                param_attr=ParamAttr(
                    name=fpn_name + "_w", initializer=Xavier(fan_out=fan)),
                bias_attr=ParamAttr(
                    name=fpn_name + "_b",
                    learning_rate=2.,
                    regularizer=L2Decay(0.)),
                name=fpn_name)
            fpn_dict[fpn_name] = fpn_output
            fpn_name_list.append(fpn_name)
        if not self.has_extra_convs and self.max_level - self.min_level == len(
                self.spatial_scale):
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
        # Coarser FPN levels introduced for RetinaNet
        highest_backbone_level = self.min_level + len(self.spatial_scale) - 1
        if self.has_extra_convs and self.max_level > highest_backbone_level:
            fpn_blob = body_dict[body_name_list[0]]
            for i in range(highest_backbone_level + 1, self.max_level + 1):
                fpn_blob_in = fpn_blob
                fpn_name = 'fpn_' + str(i)
                if i > highest_backbone_level + 1:
                    fpn_blob_in = fluid.layers.relu(fpn_blob)
                fan = fpn_blob_in.shape[1] * 3 * 3
                fpn_blob = fluid.layers.conv2d(
                    input=fpn_blob_in,
                    num_filters=self.num_chan,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    param_attr=ParamAttr(
                        name=fpn_name + "_w", initializer=Xavier(fan_out=fan)),
                    bias_attr=ParamAttr(
                        name=fpn_name + "_b",
                        learning_rate=2.,
                        regularizer=L2Decay(0.)),
                    name=fpn_name)
                fpn_dict[fpn_name] = fpn_blob
                fpn_name_list.insert(0, fpn_name)
                self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)
        res_dict = OrderedDict([(k, fpn_dict[k]) for k in fpn_name_list])
        return res_dict, self.spatial_scale


@register
class ESFPN(object):
    """
    Enhanced Single Feature Pyramid Network

    Args:
        num_chan (int): number of feature channels
        level (int) : level 
        min_level (int): lowest level of the backbone feature map to use
        max_level (int): highest level of the backbone feature map to use
        spatial_scale (list): feature map scaling factor
        has_extra_convs (bool): whether has extral convolutions in higher levels
    """

    def __init__(self,
                 num_chan=256,
                 min_level=2,
                 max_level=4,
                 spatial_scale=[1. / 16., 1. / 8., 1. / 4.],
                 has_extra_convs=False):
        self.num_chan = num_chan
        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale
        self.has_extra_convs = has_extra_convs
        return
    
    def _conv_reduce(self, input, name, num_filters=256, filter_size=1, padding=0):
        '''conv_reduce'''
        out = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=1,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + "_reduce_weights"),
            bias_attr=ParamAttr(name=name + "_reduce_biases"),
            name=name + '.reduce.output.1')
        return out
    
    def _fpn_upsampling(self, input, shape_var):
        shape_hw = fluid.layers.slice(shape_var, axes=[0], starts=[2], ends=[4])
        out_shape_ = shape_hw
        out_shape = fluid.layers.cast(out_shape_, dtype='int32')
        out_shape.stop_gradient = True
        output = fluid.layers.resize_bilinear(input, scale=2., actual_shape=out_shape)
        return output
    
    def _add_es_topdown_lateral(self, conv_upper, conv_down, output_level):
        '''
        _add_es_topdown_lateral
        conv_upper   : upper level feature
        conv_down    : low level feature
        output_level : output feature map level, same as conv_down
        '''
        down_shape_op = fluid.layers.shape(conv_down)
        up = self._fpn_upsampling(conv_upper, down_shape_op)
        
        output = fluid.layers.elementwise_add(up, conv_down)
        output_name = 'P{}.smooth.conv1_1.output.1'.format( output_level )
        output = fluid.layers.conv2d(
                    input=output,
                    num_filters=self.num_chan,
                    filter_size=3,
                    stride=1,
                    padding=1,
                    act=None,
                    param_attr=ParamAttr(name="P{}_smooth_weights".format( output_level )),
                    bias_attr=ParamAttr("P{}_smooth_biases".format( output_level )),
                    name=output_name)
        return output, output_name
    
    def conv_smooth_reduce( self, conv, level ):
        '''
        conv_smooth
        '''
        ch_med = conv.shape[1] // 2
        conv_st1 = self._conv_reduce(conv,
                                     "res{}_r1".format(level), 
                                     num_filters = ch_med,
                                     filter_size=3,
                                     padding=1)
        conv_st2 = self._conv_reduce(conv_st1,
                                     "res{}".format(level),
                                     num_filters = self.num_chan,
                                     filter_size=1,
                                     padding=0)
        return conv_st2
    
    def get_output(self, body_dict):
        body_name_list = list(body_dict.keys())[::-1]
        res5_name, res4_name, res3_name, res2_name = body_name_list

        res2 = body_dict[res2_name]
        res3 = body_dict[res3_name]
        res4 = body_dict[res4_name]
        res5 = body_dict[res5_name]
        
        #reduce two stages
        res5_reduce = self.conv_smooth_reduce( res5, 5 )
        res4_reduce = self.conv_smooth_reduce( res4, 4 )
        res3_reduce = self.conv_smooth_reduce( res3, 3 )
        
        P4, P4_name = self._add_es_topdown_lateral( res5_reduce, res4_reduce, 4 )
        P3, P3_name = self._add_es_topdown_lateral( P4, res3_reduce, 3 )
        P2, P2_name = self._add_es_topdown_lateral( P3, res2, 2 )
        
        fpn_name_list = [P4_name, P3_name, P2_name]
        fpn_dict = { P4_name:P4, P3_name:P3, P2_name:P2  }
        res_dict = OrderedDict([(k, fpn_dict[k]) for k in fpn_name_list])
        return res_dict, self.spatial_scale