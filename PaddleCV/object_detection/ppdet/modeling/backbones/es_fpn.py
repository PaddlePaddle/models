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

__all__ = ['ESFPN']


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
