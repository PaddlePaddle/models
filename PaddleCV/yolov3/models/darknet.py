#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay

def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='leaky',
                  is_test=True,
                  name=None):
    conv1 = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02),
                name=name+".conv.weights"),
        bias_attr=False)

    bn_name = name + ".bn"
    out = fluid.layers.batch_norm(
        input=conv1,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.),
                name=bn_name + '.scale'),
        bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.),
                name=bn_name + '.offset'),
        moving_mean_name=bn_name + '.mean',
        moving_variance_name=bn_name + '.var')
    if act == 'leaky':
        out = fluid.layers.leaky_relu(x=out, alpha=0.1)
    return out

def downsample(input, 
               ch_out, 
               filter_size=3, 
               stride=2, 
               padding=1, 
               is_test=True, 
               name=None):
    return conv_bn_layer(input, 
            ch_out=ch_out, 
            filter_size=filter_size, 
            stride=stride, 
            padding=padding, 
            is_test=is_test,
            name=name)

def basicblock(input, ch_out, is_test=True, name=None):
    conv1 = conv_bn_layer(input, ch_out, 1, 1, 0, 
                          is_test=is_test, name=name+".0")
    conv2 = conv_bn_layer(conv1, ch_out*2, 3, 1, 1, 
                          is_test=is_test, name=name+".1")
    out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
    return out

def layer_warp(block_func, input, ch_out, count, is_test=True, name=None):
    res_out = block_func(input, ch_out, is_test=is_test, 
                         name='{}.0'.format(name))
    for j in range(1, count):
        res_out = block_func(res_out, ch_out, is_test=is_test, 
                             name='{}.{}'.format(name, j))
    return res_out

DarkNet_cfg = {
        53: ([1,2,8,8,4],basicblock)
}

def add_DarkNet53_conv_body(body_input, is_test=True):
    stages, block_func = DarkNet_cfg[53]
    stages = stages[0:5]
    conv1 = conv_bn_layer(body_input, ch_out=32, filter_size=3, 
                          stride=1, padding=1, is_test=is_test, 
                          name="yolo_input")
    downsample_ = downsample(conv1, ch_out=conv1.shape[1]*2, 
                             is_test=is_test, 
                             name="yolo_input.downsample")
    blocks = []
    for i, stage in enumerate(stages):
        block = layer_warp(block_func, downsample_, 32 *(2**i), 
                           stage, is_test=is_test, 
                           name="stage.{}".format(i))
        blocks.append(block)
        if i < len(stages) - 1: # do not downsaple in the last stage
            downsample_ = downsample(block, ch_out=block.shape[1]*2, 
                                     is_test=is_test, 
                                     name="stage.{}.downsample".format(i))
    return blocks[-1:-4:-1]

