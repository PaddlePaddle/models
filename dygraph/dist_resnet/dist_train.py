 Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import time

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable
from reader import train, val

batch_size = 32
epoch = 1

total_images = 1281167

def optimizer_setting():
    optimizer = fluid.optimizer.Momentum(learning_rate=0.1,momentum=0.9)
    return optimizer


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=None)

        self._batch_norm = BatchNorm(self.full_name(), num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__(name_scope)

        self.conv0 = ConvBNLayer(
            self.full_name(),
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            self.full_name(),
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                self.full_name(),
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=conv2)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)


class ResNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50, class_dim=102):
        super(ResNet, self).__init__(name_scope)

        self.layers = layers
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

        self.conv = ConvBNLayer(
            self.full_name(),
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = Pool2D(
            self.full_name(),
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        self.full_name(),
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            self.full_name(), pool_size=7, pool_type='avg', global_pooling=True)

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.fc= FC(self.full_name(),
                      size=class_dim,
                      act='softmax',
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))
 
    def forward(self, inputs, label):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = self.fc(y)
        acc1 = fluid.layers.accuracy(input=y, label=label, k=1)
        acc5 = fluid.layers.accuracy(input=y, label=label, k=5)

        return y, acc1, acc5

def train_resnet():
    seed = 90
    place = fluid.CUDAPlace(dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        np.random.seed(seed)
        import random
        random.seed = seed

        resnet = ResNet("dist_resnet", class_dim=1000)
        strategy = dygraph.parallel.ParallelStrategy()
        strategy.nranks = dygraph.parallel.Env().nranks
        strategy.local_rank = dygraph.parallel.Env().local_rank
        strategy.trainer_endpoints = dygraph.parallel.Env().trainer_endpoints
        strategy.current_endpoint = dygraph.parallel.Env().current_endpoint 
        resnet = dygraph.parallel.DataParallel(resnet, strategy)
        if strategy.nranks > 1:
            dygraph.parallel.prepare_context(strategy)
        
        optimizer = optimizer_setting()
        train_reader = paddle.batch(
            train(data_dir="/imagenet/ImageNet_resize/",
                  pass_id_as_seed=0, infinite=True),
            batch_size=batch_size, drop_last=True)
        steps_per_epoch = int(total_images / strategy.nranks / batch_size)
        print("steps per eoch: %d" % steps_per_epoch)
        for eop in range(epoch):
            for step_id, data in enumerate(train_reader()):
                if step_id == steps_per_epoch:
                    break
                if len(np.array([x[1] for x in data]).astype('int64')) != batch_size:
                    continue

                s_time = time.time()
                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    batch_size, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label._stop_gradient = True
   
                out, acc1, acc5 = resnet(img, label)

                loss = fluid.layers.cross_entropy(input=out, label=label)
                avg_loss = fluid.layers.mean(x=loss)
                dy_out = avg_loss.numpy()
                avg_loss = resnet.scale_loss(avg_loss)
                avg_loss.backward()
                resnet.apply_collective_grads()
                optimizer.minimize(avg_loss)
                throughtput = batch_size / (time.time() - s_time)
                print("epoch id: %d, step: %d, loss: %f, acc1: %f, acc5: %f, throughtput: %f imgs/s " %
                        (eop, step_id, dy_out, float(acc1.numpy()), float(acc5.numpy()), throughtput))
                resnet.clear_gradients()

if __name__ == '__main__':
    train_resnet()
