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
from config import cfg

def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='leaky',
                  i=0):
    conv1 = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02),
                name="conv" + str(i)+"_weights"),
        bias_attr=False)

    bn_name = "bn" + str(i)

    out = fluid.layers.batch_norm(
        input=conv1,
        act=None,
        is_test=True,
        param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.),
                name=bn_name + '_scale'),
        bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.),
                name=bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_var')
    if act == 'leaky':
        out = fluid.layers.leaky_relu(x=out, alpha=0.1)
    return out

def basicblock(input, ch_out, stride,i):
    """
    channel: convolution channels for 1x1 conv
    """
    conv1 = conv_bn_layer(input, ch_out, 1, 1, 0, i=i)
    conv2 = conv_bn_layer(conv1, ch_out*2, 3, 1, 1, i=i+1)
    out = fluid.layers.elementwise_add(x=input, y=conv2, act=None,name="res"+str(i+2))
    return out

def layer_warp(block_func, input, ch_out, count, stride,i):
    res_out = block_func(input, ch_out, stride, i=i)
    for j in range(1, count):
        res_out = block_func(res_out, ch_out, 1 ,i=i+j*3)
    return res_out

DarkNet_cfg = {
        53: ([1,2,8,8,4],basicblock)
}

# num_filters = [32, 64, 128, 256, 512, 1024]

def add_DarkNet53_conv_body(body_input):

    stages, block_func = DarkNet_cfg[53]
    stages = stages[0:5]
    conv1 = conv_bn_layer(
            body_input, ch_out=32, filter_size=3, stride=1, padding=1, act="leaky",i=0)
    conv2 = conv_bn_layer(
            conv1, ch_out=64, filter_size=3, stride=2, padding=1, act="leaky", i=1)
    block3 = layer_warp(block_func, conv2, 32, stages[0], 1, i=2)
    downsample3 = conv_bn_layer(
            block3, ch_out=128, filter_size=3, stride=2, padding=1, i=5)
    block4 = layer_warp(block_func, downsample3, 64, stages[1], 1, i=6)
    downsample4 = conv_bn_layer(
            block4, ch_out=256, filter_size=3, stride=2, padding=1, i=12)
    block5 = layer_warp(block_func, downsample4, 128, stages[2], 1,i=13)
    downsample5 = conv_bn_layer(
            block5, ch_out=512, filter_size=3, stride=2, padding=1, i=37)
    block6 = layer_warp(block_func, downsample5, 256, stages[3], 1, i=38)
    downsample6 = conv_bn_layer(
            block6, ch_out=1024, filter_size=3, stride=2, padding=1,  i=62)
    block7 = layer_warp(block_func, downsample6, 512, stages[4], 1,i=63)
    return block7,block6,block5

