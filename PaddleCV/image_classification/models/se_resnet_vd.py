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
from paddle.fluid.param_attr import ParamAttr
import math

__all__ = ["SE_ResNet_vd", "SE_ResNet18_vd","SE_ResNet34_vd", "SE_ResNet50_vd", "SE_ResNet101_vd", "SE_ResNet152_vd", 
           "SE_ResNet200_vd"]


class SE_ResNet_vd():
    def __init__(self, layers=50, is_3x3=False):
        self.layers = layers
        self.is_3x3 = is_3x3

    def net(self, input, class_dim=1000):
        is_3x3 = self.is_3x3
        layers = self.layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]     
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_filters = [64, 128, 256, 512]
        reduction_ratio = 16
        if is_3x3 == False:
            conv = self.conv_bn_layer(
                input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        else:
            conv = self.conv_bn_layer(
                input=input, num_filters=32, filter_size=3, stride=2, act='relu', name='conv1_1')
            conv = self.conv_bn_layer(
                input=conv, num_filters=32, filter_size=3, stride=1, act='relu', name='conv1_2')
            conv = self.conv_bn_layer(
                input=conv, num_filters=64, filter_size=3, stride=1, act='relu', name='conv1_3')

        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152, 200] and block == 2:
                        if i == 0:
                            conv_name="res"+str(block+2)+"a"
                        else:
                            conv_name="res"+str(block+2)+"b"+str(i)
                    else:
                        conv_name="res"+str(block+2)+chr(97+i)
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        if_first=block==i==0, 
                        reduction_ratio=reduction_ratio,
                        name=conv_name)

        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name="res"+str(block+2)+chr(97+i)
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        if_first=block==i==0,
                        reduction_ratio=reduction_ratio,
                        name=conv_name)

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)


        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv, stdv), name='fc6_weights'),
                              bias_attr=ParamAttr(name='fc6_offset'))
        
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
            padding=(filter_size - 1) // 2,
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
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance')
    
    
    def conv_bn_layer_new(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        pool = fluid.layers.pool2d(input=input,
            pool_size=2,
            pool_stride=2,
            pool_padding=0,
            pool_type='avg',
            ceil_mode=True)
        
        conv = fluid.layers.conv2d(
            input=pool,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
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
                                       param_attr=ParamAttr(name=bn_name + '_scale'),
                                       bias_attr=ParamAttr(bn_name + '_offset'),
                                       moving_mean_name=bn_name + '_mean',
                                       moving_variance_name=bn_name + '_variance')
    
    

    def shortcut(self, input, ch_out, stride, name, if_first=False):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            if if_first:
                return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
            else:
                return self.conv_bn_layer_new(input, ch_out, 1, stride, name=name)
        elif if_first:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name, if_first, reduction_ratio):
        conv0 = self.conv_bn_layer(
            input=input, 
            num_filters=num_filters, 
            filter_size=1, 
            act='relu', 
            name=name+"_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name+"_branch2b")
        conv2 =self.conv_bn_layer(
            input=conv1, 
            num_filters=num_filters * 4, 
            filter_size=1, 
            act=None, 
            name=name+"_branch2c")
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 4,
            reduction_ratio=reduction_ratio,
            name='fc_'+name)

        short = self.shortcut(input, num_filters * 4, stride, if_first=if_first, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')
    
    def basic_block(self, input, num_filters, stride, name, if_first, reduction_ratio):
        conv0 = self.conv_bn_layer(input=input, 
                                   num_filters=num_filters, 
                                   filter_size=3, 
                                   act='relu', 
                                   stride=stride,
                                   name=name+"_branch2a")
        conv1 = self.conv_bn_layer(input=conv0, 
                                   num_filters=num_filters, 
                                   filter_size=3, 
                                   act=None, 
                                   name=name+"_branch2b")
        scale = self.squeeze_excitation(
            input=conv1,
            num_channels=num_filters,
            reduction_ratio=reduction_ratio,
            name='fc_'+name)
        short = self.shortcut(input, 
                              num_filters, 
                              stride, 
                              if_first=if_first, 
                              name=name + "_branch1")
        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')
    
    
    def squeeze_excitation(self, input, num_channels, reduction_ratio, name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(input=pool,
                                  size=num_channels // reduction_ratio,
                                  act='relu',
                                  param_attr=fluid.param_attr.ParamAttr(
                                      initializer=fluid.initializer.Uniform(
                                          -stdv, stdv),name=name+'_sqz_weights'),
                                 bias_attr=ParamAttr(name=name+'_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(input=squeeze,
                                     size=num_channels,
                                     act='sigmoid',
                                     param_attr=fluid.param_attr.ParamAttr(
                                         initializer=fluid.initializer.Uniform(-stdv, stdv),
                                         name=name+'_exc_weights'),
                                    bias_attr=ParamAttr(name=name+'_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale

def SE_ResNet18_vd():
    model = SE_ResNet_vd(layers=18, is_3x3 = True)
    return model

def SE_ResNet34_vd():
    model = SE_ResNet_vd(layers=34, is_3x3 = True)
    return model
    
def SE_ResNet50_vd():
    model = SE_ResNet_vd(layers=50, is_3x3 = True)
    return model

def SE_ResNet101_vd():
    model = SE_ResNet_vd(layers=101, is_3x3 = True)
    return model

def SE_ResNet152_vd():
    model = SE_ResNet_vd(layers=152, is_3x3 = True)
    return model

def SE_ResNet200_vd():
    model = SE_ResNet_vd(layers=200, is_3x3 = True)
    return model

