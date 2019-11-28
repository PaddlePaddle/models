#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import math
from paddle.fluid.param_attr import ParamAttr

__all__ = ["Res2Net", "Res2Net50_48w_2s", "Res2Net50_26w_4s", "Res2Net50_14w_8s", "Res2Net50_26w_6s", "Res2Net50_26w_8s", 
           "Res2Net101_26w_4s", "Res2Net152_26w_4s"]


class Res2Net():
    
    def __init__(self, layers=50, scales=4, width=26):
        self.layers = layers
        self.scales = scales
        self.width = width   

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        basic_width = self.width * self.scales
        num_filters1 = [basic_width * t for t in [1, 2, 4, 8]]
        num_filters2 = [256 * t for t in [1, 2, 4, 8]]
        
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu', name="conv1")
        
        
        conv = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block+2) + "a"
                    else:
                        conv_name = "res" + str(block+2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block+2) + chr(97+i)
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters1=num_filters1[block],
                    num_filters2=num_filters2[block],
                    stride=2 if i==0 and block !=0 else 1, name=conv_name)
        pool = fluid.layers.pool2d(
                input=conv, pool_size=7, pool_stride=1, pool_type='avg', global_pooling=True)
        
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),name='fc_weights'),
            bias_attr=fluid.param_attr.ParamAttr(name='fc_offset'))
        return out
    

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1)//2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:] 
        
        return fluid.layers.batch_norm(input=conv, 
                                       act=act,
                                       param_attr=ParamAttr(name=bn_name+'_scale'),
                                       bias_attr=ParamAttr(bn_name+'_offset'),
                                       moving_mean_name=bn_name+'_mean',
                                       moving_variance_name=bn_name+'_variance')
        
        
    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input


    def bottleneck_block(self, input, num_filters1, num_filters2, stride, name):
        conv0 = self.conv_bn_layer(
            input=input, 
            num_filters=num_filters1, 
            filter_size=1, 
            stride=1, 
            act='relu', 
            name=name+'_branch2a')
        xs = fluid.layers.split(conv0, self.scales, 1)
        ys = []
        for s in range(self.scales - 1):
            if s == 0 or stride == 2:
                ys.append(self.conv_bn_layer(input=xs[s], 
                                             num_filters=num_filters1//self.scales, 
                                             stride=stride, 
                                             filter_size=3, 
                                             act='relu', 
                                             name=name+'_branch2b_'+str(s+1)))
            else:
                ys.append(self.conv_bn_layer(input=xs[s]+ys[-1], 
                                             num_filters=num_filters1//self.scales, 
                                             stride=stride, 
                                             filter_size=3, 
                                             act='relu', 
                                             name=name+'_branch2b_'+str(s+1))) 
        if stride == 1:
            ys.append(xs[-1])
        else:
            ys.append(fluid.layers.pool2d(input=xs[-1], 
                                          pool_size=3, 
                                          pool_stride=stride, 
                                          pool_padding=1, 
                                          pool_type='avg'))

        conv1 = fluid.layers.concat(ys, axis=1)
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters2, filter_size=1, act=None, name=name+"_branch2c")

        short = self.shortcut(input, num_filters2, stride, name=name+"_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')        



def Res2Net50_48w_2s():
    model = Res2Net(layers=50, scales=2, width=48)
    return model


def Res2Net50_26w_4s():
    model = Res2Net(layers=50, scales=4, width=26)
    return model


def Res2Net50_14w_8s():
    model = Res2Net(layers=50, scales=8, width=14)
    return model


def Res2Net50_26w_6s():
    model = Res2Net(layers=50, scales=6, width=26)
    return model


def Res2Net50_26w_8s():
    model = Res2Net(layers=50, scales=8, width=26)
    return model


def Res2Net101_26w_4s():
    model = Res2Net(layers=101, scales=4, width=26)
    return model


def Res2Net152_26w_4s():
    model = Res2Net(layers=152, scales=4, width=26)
    return model
