from __future__ import division
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D,  Conv2DTranspose , BatchNorm ,Pool2D
import os

# cudnn is not better when batch size is 1.
use_cudnn = False


class conv2d(fluid.dygraph.Layer):
    """docstring for Conv2D"""
    def __init__(self, 
                name_scope,
                num_filters=64,
                filter_size=7,
                stride=1,
                stddev=0.02,
                padding=0,
                norm=True,
                relu=True,
                relufactor=0.0,
                use_bias=False):
        super(conv2d, self).__init__(name_scope)

        if use_bias == False:
            con_bias_attr = False
        else:
            con_bias_attr = fluid.ParamAttr(name="conv_bias",initializer=fluid.initializer.Constant(0.0))

        self.conv = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            param_attr=fluid.ParamAttr(
                name="conv2d_weights",
                initializer=fluid.initializer.NormalInitializer(loc=0.0,scale=stddev)),
            bias_attr=con_bias_attr)
        if norm:
            self.bn = BatchNorm(self.full_name(),
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    name="scale",
                    initializer=fluid.initializer.NormalInitializer(1.0,0.02)),
                bias_attr=fluid.ParamAttr(
                    name="bias",
                    initializer=fluid.initializer.Constant(0.0)),
                trainable_statistics=True
                )
    
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu

    
    def forward(self,inputs):
        conv = self.conv(inputs)
        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
        return conv


class DeConv2D(fluid.dygraph.Layer):
    def __init__(self,
            name_scope,
            num_filters=64,
            filter_size=7,
            stride=1,
            stddev=0.02,
            padding=[0,0],
            outpadding=[0,0,0,0],
            relu=True,
            norm=True,
            relufactor=0.0,
            use_bias=False
            ):
        super(DeConv2D,self).__init__(name_scope)

        if use_bias == False:
            de_bias_attr = False
        else:
            de_bias_attr = fluid.ParamAttr(name="de_bias",initializer=fluid.initializer.Constant(0.0))

        self._deconv = Conv2DTranspose(self.full_name(),
                                        num_filters,
                                        filter_size=filter_size,
                                        stride=stride,
                                        padding=padding,
                                        param_attr=fluid.ParamAttr(
                                            name="this_is_deconv_weights",
                                            initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=stddev)),
                                        bias_attr=de_bias_attr)



        if norm:
            self.bn = BatchNorm(self.full_name(),
                num_channels=num_filters,
                param_attr=fluid.ParamAttr(
                    name="de_wights",
                    initializer=fluid.initializer.NormalInitializer(1.0, 0.02)),
                bias_attr=fluid.ParamAttr(name="de_bn_bias",initializer=fluid.initializer.Constant(0.0)),
                trainable_statistics=True)        
        self.outpadding = outpadding
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu

    def forward(self,inputs):
        #todo: add use_bias
        #if self.use_bias==False:
        conv = self._deconv(inputs)
        #else:
        #    conv = self._deconv(inputs)
        conv = fluid.layers.pad2d(conv, paddings=self.outpadding, mode='constant', pad_value=0.0)

        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
        return conv
