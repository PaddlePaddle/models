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

import os
import time
import sys
import paddle.fluid as fluid
import math


class StNet_ResNet():
    def __init__(self, layers=50, seg_num=7, seglen=5, is_training=True):
        self.layers = layers
        self.seglen = seglen
        self.seg_num = seg_num
        self.is_training = is_training

    def temporal_conv_bn(
            self,
            input,  #(B*seg_num, c, h, w)
            num_filters,
            filter_size=(3, 1, 1),
            padding=(1, 0, 0)):
        #(B, seg_num, c, h, w)
        in_reshape = fluid.layers.reshape(
            x=input,
            shape=[
                -1, self.seg_num, input.shape[-3], input.shape[-2],
                input.shape[-1]
            ])
        in_transpose = fluid.layers.transpose(in_reshape, perm=[0, 2, 1, 3, 4])

        conv = fluid.layers.conv3d(
            input=in_transpose,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=1,
            groups=1,
            padding=padding,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.MSRAInitializer()),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0.0)))

        out = fluid.layers.batch_norm(
            input=conv,
            act=None,
            is_test=(not self.is_training),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=1.0)),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0.0)))
        out = out + in_transpose
        out = fluid.layers.transpose(out, perm=[0, 2, 1, 3, 4])
        out = fluid.layers.reshape(x=out, shape=input.shape)
        return out

    def xception(self, input):  #(B, C, seg_num,1)
        bn = fluid.layers.batch_norm(
            input=input,
            act=None,
            name="xception_bn",
            is_test=(not self.is_training),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=1.0)),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0.0)))
        att_conv = fluid.layers.conv2d(
            input=bn,
            num_filters=2048,
            filter_size=[3, 1],
            stride=[1, 1],
            padding=[1, 0],
            groups=2048,
            name="xception_att_conv",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.MSRAInitializer()),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0)))
        att_2 = fluid.layers.conv2d(
            input=att_conv,
            num_filters=1024,
            filter_size=[1, 1],
            stride=[1, 1],
            name="xception_att_2",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.MSRAInitializer()),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0)))
        bndw = fluid.layers.batch_norm(
            input=att_2,
            act="relu",
            name="xception_bndw",
            is_test=(not self.is_training),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=1.0)),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0.0)))
        att1 = fluid.layers.conv2d(
            input=bndw,
            num_filters=1024,
            filter_size=[3, 1],
            stride=[1, 1],
            padding=[1, 0],
            groups=1024,
            name="xception_att1",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.MSRAInitializer()),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0)))
        att1_2 = fluid.layers.conv2d(
            input=att1,
            num_filters=1024,
            filter_size=[1, 1],
            stride=[1, 1],
            name="xception_att1_2",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.MSRAInitializer()),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0)))
        dw = fluid.layers.conv2d(
            input=bn,
            num_filters=1024,
            filter_size=[1, 1],
            stride=[1, 1],
            name="xception_dw",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.MSRAInitializer()),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0)))
        add_to = dw + att1_2
        bn2 = fluid.layers.batch_norm(
            input=add_to,
            act=None,
            name='xception_bn2',
            is_test=(not self.is_training),
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=1.0)),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(value=0.0)))
        return fluid.layers.relu(bn2)

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
            param_attr=fluid.param_attr.ParamAttr(name=name + "_weights"),
            bias_attr=False,
            #name = name+".conv2d.output.1"
        )
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            is_test=(not self.is_training),
            #name=bn_name+'.output.1',
            param_attr=fluid.param_attr.ParamAttr(name=bn_name + "_scale"),
            bias_attr=fluid.param_attr.ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + '_variance')

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self.shortcut(
            input, num_filters * 4, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(
            x=short,
            y=conv2,
            act='relu',
            #name=".add.output.5"
        )

    def net(self, input, class_dim=101):
        layers = self.layers
        seg_num = self.seg_num
        seglen = self.seglen

        supported_layers = [50, 101, 152]
        if layers not in supported_layers:
            print("supported layers are", supported_layers, \
                  "but input layer is ", layers)
            exit()

        # reshape input
        # [B, seg_num, seglen*c, H, W] --> [B*seg_num, seglen*c, H, W]
        channels = input.shape[2]
        short_size = input.shape[3]
        input = fluid.layers.reshape(
            x=input, shape=[-1, channels, short_size, short_size])

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu',
            name='conv1')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                if layers in [101, 152] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block + 2) + "a"
                    else:
                        conv_name = "res" + str(block + 2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block + 2) + chr(97 + i)

                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    name=conv_name)
            if block == 1:
                #insert the first temporal modeling block
                conv = self.temporal_conv_bn(input=conv, num_filters=512)
            if block == 2:
                #insert the second temporal modeling block
                conv = self.temporal_conv_bn(input=conv, num_filters=1024)

        pool = fluid.layers.pool2d(
            input=conv, pool_size=7, pool_type='avg', global_pooling=True)

        feature = fluid.layers.reshape(
            x=pool, shape=[-1, seg_num, pool.shape[1], 1])
        feature = fluid.layers.transpose(feature, perm=[0, 2, 1, 3])

        #append the temporal Xception block
        xfeat = self.xception(feature)  #(B, 1024, seg_num, 1)
        out = fluid.layers.pool2d(
            input=xfeat,
            pool_size=(seg_num, 1),
            pool_type='max',
            global_pooling=True)
        out = fluid.layers.reshape(x=out, shape=[-1, 1024])

        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=out,
                              size=class_dim,
                              act='softmax',
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Uniform(-stdv,
                                                                        stdv)))
        return out
