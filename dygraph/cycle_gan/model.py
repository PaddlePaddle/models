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

from layers import *
import paddle.fluid as fluid


class build_resnet_block(fluid.dygraph.Layer):
    def __init__(self,
        dim,
        use_bias=False):
        super(build_resnet_block,self).__init__()

        self.conv0 = conv2d(
            num_channels=dim,
            num_filters=dim,
            filter_size=3,
            stride=1,
            stddev=0.02,
            use_bias=False)
        self.conv1 = conv2d(
            num_channels=dim,
            num_filters=dim,
            filter_size=3,
            stride=1,
            stddev=0.02,
            relu=False,
            use_bias=False)
        self.dim = dim
    def forward(self,inputs):
        out_res = fluid.layers.pad2d(inputs, [1, 1, 1, 1], mode="reflect")
        out_res = self.conv0(out_res)
        
        #if self.use_dropout:
        #    out_res = fluid.layers.dropout(out_res,dropout_prod=0.5)
        out_res = fluid.layers.pad2d(out_res, [1, 1, 1, 1], mode="reflect")
        out_res = self.conv1(out_res)
        return out_res + inputs


class build_generator_resnet_9blocks(fluid.dygraph.Layer):
    def __init__ (self, input_channel):
        super(build_generator_resnet_9blocks, self).__init__()

        self.conv0 = conv2d(
            num_channels=input_channel,
            num_filters=32,
            filter_size=7,
            stride=1,
            padding=0,
            stddev=0.02)
        self.conv1 = conv2d(
            num_channels=32,
            num_filters=64,
            filter_size=3,
            stride=2,
            padding=1,
            stddev=0.02)
        self.conv2 = conv2d(
            num_channels=64,
            num_filters=128,
            filter_size=3,
            stride=2,
            padding=1,
            stddev=0.02)
        self.build_resnet_block_list=[]
        dim = 128
        for i in range(9):
            Build_Resnet_Block = self.add_sublayer(
                "generator_%d" % (i+1),
                build_resnet_block(dim))
            self.build_resnet_block_list.append(Build_Resnet_Block)
        self.deconv0 = DeConv2D(
            num_channels=dim,
            num_filters=32*2,
            filter_size=3,
            stride=2,
            stddev=0.02,
            padding=[1, 1],
            outpadding=[0, 1, 0, 1],
            )
        self.deconv1 = DeConv2D(
            num_channels=32*2,
            num_filters=32,
            filter_size=3,
            stride=2,
            stddev=0.02,
            padding=[1, 1],
            outpadding=[0, 1, 0, 1])
        self.conv3 = conv2d(
            num_channels=32,
            num_filters=input_channel,
            filter_size=7,
            stride=1,
            stddev=0.02,
            padding=0,
            relu=False,
            norm=False,
            use_bias=True)

    def forward(self,inputs):
        pad_input = fluid.layers.pad2d(inputs, [3, 3, 3, 3], mode="reflect")
        y = self.conv0(pad_input)
        y = self.conv1(y)
        y = self.conv2(y)
        for build_resnet_block_i in self.build_resnet_block_list:
            y = build_resnet_block_i(y)
        y = self.deconv0(y)
        y = self.deconv1(y)
        y = fluid.layers.pad2d(y,[3,3,3,3],mode="reflect")
        y = self.conv3(y)
        y = fluid.layers.tanh(y)
        return y


class build_gen_discriminator(fluid.dygraph.Layer):
    def __init__(self, input_channel):
        super(build_gen_discriminator, self).__init__()
        
        self.conv0 = conv2d(
            num_channels=input_channel,
            num_filters=64,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            norm=False,
            use_bias=True,
            relufactor=0.2)
        self.conv1 = conv2d(
            num_channels=64,
            num_filters=128,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            relufactor=0.2)
        self.conv2 = conv2d(
            num_channels=128,
            num_filters=256,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            relufactor=0.2)
        self.conv3 = conv2d(
            num_channels=256,
            num_filters=512,
            filter_size=4,
            stride=1,
            stddev=0.02,
            padding=1,
            relufactor=0.2)
        self.conv4 = conv2d(
            num_channels=512,
            num_filters=1,
            filter_size=4,
            stride=1,
            stddev=0.02,
            padding=1,
            norm=False,
            relu=False,
            use_bias=True)

    def forward(self,inputs):
        y = self.conv0(inputs)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        return y


