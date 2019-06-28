# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import unittest
import numpy as np
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable
import sys
import math

batch_size = 64
train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "cosine_decay",
        "batch_size": batch_size,
        "epochs": [40, 80, 100],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    },
    "batch_size": batch_size,
    "lr": 0.0125,
    "total_images": 6149,
    "num_epochs":200
}

momentum_rate = 0.9
l2_decay = 1.2e-4

def optimizer_setting(params):
    ls = params["learning_strategy"]
    if "total_images" not in params:
        total_images = 6149
    else:
        total_images = params["total_images"]
    
    batch_size = ls["batch_size"]
    step = int(math.ceil(float(total_images) / batch_size))
    bd = [step * e for e in ls["epochs"]]
    lr = params["lr"]
    num_epochs = params["num_epochs"]
    print("lr:",lr)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.cosine_decay(
            learning_rate=lr,step_each_epoch=step,epochs=num_epochs),
        momentum=momentum_rate,
        regularization=fluid.regularizer.L2Decay(l2_decay))

    return optimizer


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            "conv2d",
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
	    param_attr=fluid.ParamAttr(name="weights"))

        self._batch_norm = BatchNorm(self.full_name(), num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class SqueezeExcitation(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_channels, reduction_ratio):

        super(SqueezeExcitation, self).__init__(name_scope)
        self._pool = Pool2D(
            self.full_name(), pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0/math.sqrt(num_channels*1.0)
        self._squeeze = FC(
            self.full_name(),
            size=num_channels // reduction_ratio,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv,stdv)),
            act='relu')
        stdv = 1.0/math.sqrt(num_channels/16.0*1.0)
        self._excitation = FC(
            self.full_name(),
            size=num_channels,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv,stdv)),
            act='sigmoid')
    def forward(self, input):
        y = self._pool(input)
        y = self._squeeze(y)
        y = self._excitation(y)
        y = fluid.layers.elementwise_mul(x=input, y=y, axis=0)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 stride,
                 cardinality,
                 reduction_ratio,
                 shortcut=True):
        super(BottleneckBlock, self).__init__(name_scope)

        self.conv0 = ConvBNLayer(
            self.full_name(),
            num_filters=num_filters,
            filter_size=1,
            act="relu")
        self.conv1 = ConvBNLayer(
            self.full_name(),
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act="relu")
        self.conv2 = ConvBNLayer(
            self.full_name(),
            num_filters=num_filters * 2,
            filter_size=1,
            act=None)

        self.scale = SqueezeExcitation(
            self.full_name(),
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio)

        if not shortcut:
            self.short = ConvBNLayer(
                self.full_name(),
                num_filters=num_filters * 2,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 2

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = fluid.layers.elementwise_add(x=short, y=scale, act='relu')
        return y


class SeResNeXt(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50, class_dim=102):
        super(SeResNeXt, self).__init__(name_scope)

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                self.full_name(),
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            self.pool = Pool2D(
                self.full_name(),
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                self.full_name(),
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
            self.pool = Pool2D(
                self.full_name(),
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                self.full_name(),
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu')
            self.conv1 = ConvBNLayer(
                self.full_name(),
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu')
            self.conv2 = ConvBNLayer(
                self.full_name(),
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu')
            self.pool = Pool2D(
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
                        cardinality=cardinality,
                        reduction_ratio=reduction_ratio,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            self.full_name(), pool_size=7, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = FC(self.full_name(),
                      size=class_dim,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        if self.layers == 50 or self.layers == 101:
            y = self.conv0(inputs)
            y = self.pool(y)
        elif self.layers == 152:
            y = self.conv0(inputs)
            y = self.conv1(inputs)
            y = self.conv2(inputs)
            y = self.pool(y)

        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.dropout(y, dropout_prob=0.5,seed=100)
        y = self.out(y)
        return y


def eval(model, data):

    model.eval()
    batch_size=32
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0
    for batch_id, data in enumerate(data()):
        dy_x_data = np.array(
            [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
        if len(np.array([x[1] for x in data]).astype('int64')) != batch_size:
            continue
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(
            batch_size, 1)

        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label._stop_gradient = True
        out = model(img)
        cost,pred = fluid.layers.softmax_with_cross_entropy(out,label,return_softmax=True)
        avg_loss = fluid.layers.mean(x=cost)

        acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

        dy_out = avg_loss.numpy()

        total_loss += dy_out
        total_acc1 += acc_top1.numpy()
        total_acc5 += acc_top5.numpy()
        total_sample += 1
        if batch_id % 10 == 0:
            print("test | batch step %d, loss %0.3f acc1 %0.3f acc5 %0.3f" % \
                  ( batch_id, total_loss / total_sample, \
                   total_acc1 / total_sample, total_acc5 / total_sample))
	    
    print("final eval loss %0.3f acc1 %0.3f acc5 %0.3f" % \
          (total_loss / total_sample, \
           total_acc1 / total_sample, total_acc5 / total_sample))

def train():
    seed = 90
    epoch_num = train_parameters["num_epochs"]

    batch_size = train_parameters["batch_size"]

    with fluid.dygraph.guard():
        fluid.default_startup_program().random_seed = 90
        fluid.default_main_program().random_seed = 90
        
        se_resnext = SeResNeXt("se_resnext")
        optimizer = optimizer_setting(train_parameters)
        train_reader = paddle.batch(
            paddle.dataset.flowers.train(use_xmap=False),
            batch_size=batch_size,
            drop_last=True
            )
        
        test_reader = paddle.batch(
            paddle.dataset.flowers.test(use_xmap=False), batch_size=32)       

        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        for epoch_id in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                
                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224)
                    for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(
                        batch_size, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                out = se_resnext(img)
                softmax_out = fluid.layers.softmax(out,use_cudnn=False)
                loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
                avg_loss = fluid.layers.mean(x=loss)
                
                acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=softmax_out, label=label, k=5)

                dy_out = avg_loss.numpy()
                avg_loss.backward()

                optimizer.minimize(avg_loss)
                se_resnext.clear_gradients()
                
                lr = optimizer._global_learning_rate().numpy()
                total_loss += dy_out
                total_acc1 += acc_top1.numpy()
                total_acc5 += acc_top5.numpy()
                total_sample += 1
                if batch_id % 10 == 0:
                    print( "epoch %d | batch step %d, loss %0.3f acc1 %0.3f acc5 %0.3f lr %0.5f" % \
                           ( epoch_id, batch_id, total_loss / total_sample, \
                             total_acc1 / total_sample, total_acc5 / total_sample, lr))

            print("epoch %d | batch step %d, loss %0.3f acc1 %0.3f acc5 %0.3f" % \
                  (epoch_id, batch_id, total_loss / total_sample, \
                   total_acc1 / total_sample, total_acc5 / total_sample))
            se_resnext.eval()
            eval(se_resnext, test_reader)
            se_resnext.train()

if __name__ == '__main__':
    train()
