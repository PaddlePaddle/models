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
import paddle
import paddle.fluid as fluid
import math
__all__ = ["ResNet_vd", "ResNet50_vd_v0_embedding","ResNet101_vd_v0_embedding", "ResNet152_vd_v0_embedding"]
train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}
class ResNet_vd():
    def __init__(self, layers=50, is_3x3 = False):
        self.params = train_parameters
        self.layers = layers
        self.is_3x3 = is_3x3
    def net(self, input, embedding_size=256):
        is_3x3 = self.is_3x3
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]
        endpoints = {}
        if is_3x3 == False:
            conv = self.conv_bn_layer(
                input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        else:
            conv = self.conv_bn_layer(
                input=input, num_filters=32, filter_size=3, stride=2, act='relu')
            conv = self.conv_bn_layer(
                input=conv, num_filters=32, filter_size=3, stride=1, act='relu')
            conv = self.conv_bn_layer(
                input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    if_first=block==0)
        pool = fluid.layers.pool2d(
            input=conv, pool_size=14, pool_type='avg', global_pooling=True)
        if embedding_size > 0:
            embedding = fluid.layers.fc(input=pool, size=embedding_size)
            endpoints['embedding'] = embedding
        else:
            endpoints['embedding'] = pool
        return endpoints
 
    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) / 2,
            groups=groups,
            act=None,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)
    
    def conv_bn_layer_new(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None):
        pool = fluid.layers.pool2d(input=input,
            pool_size=2,
            pool_stride=2,
            pool_padding=0,
            pool_type='avg')
        
        conv = fluid.layers.conv2d(
            input=pool,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=1,
            padding=(filter_size - 1) / 2,
            groups=groups,
            act=None,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)
    
    def shortcut(self, input, ch_out, stride, if_first=False):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            if if_first:
                return self.conv_bn_layer(input, ch_out, 1, stride)
            else:
                return self.conv_bn_layer_new(input, ch_out, 1, stride)
        else:
            return input
    def bottleneck_block(self, input, num_filters, stride, if_first):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None)
        short = self.shortcut(input, num_filters * 4, stride, if_first=if_first)
        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')
    
def ResNet50_vd_v0_embedding():
    model = ResNet_vd(layers=50, is_3x3 = True)
    return model
def ResNet101_vd_v0_embedding():
    model = ResNet_vd(layers=101, is_3x3 = True)
    return model
def ResNet152_vd_v0_embedding():
    model = ResNet_vd(layers=152, is_3x3 = True)
    return model