# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import time
import sys
import sys
import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable

from paddle.fluid import framework

import math
import sys
import reader
from utils import *

c_scale = 1.0
IMAGENET1000 = 1281167
base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4

args = parse_args()
if int(os.getenv("PADDLE_TRAINER_ID", 0)) == 0:
    print_arguments(args)


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 name=None,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__(name)

        tmp_param = ParamAttr(name=name + "_weights")
        self._conv = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=tmp_param,
            bias_attr=False)

        self._batch_norm = BatchNorm(
            self.full_name(),
            num_filters,
            param_attr=ParamAttr(name=name + "_bn" + "_scale"),
            bias_attr=ParamAttr(name=name + "_bn" + "_offset"),
            moving_mean_name=name + "_bn" + '_mean',
            moving_variance_name=name + "_bn" + '_variance')

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if if_act:
            y = fluid.layers.relu6(y)
        return y


class InvertedResidualUnit(fluid.dygraph.Layer):
    def __init__(self,
                 num_in_filter,
                 num_filters,
                 stride,
                 filter_size,
                 padding,
                 expansion_factor,
                 name=None):
        super(InvertedResidualUnit, self).__init__(name)
        num_expfilter = int(round(num_in_filter * expansion_factor))
        self._expand_conv = ConvBNLayer(
            name=name + "_expand",
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

        self._bottleneck_conv = ConvBNLayer(
            name=name + "_dwise",
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            use_cudnn=False)

        self._linear_conv = ConvBNLayer(
            name=name + "_linear",
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1)

    def forward(self, inputs, ifshortcut):
        y = self._expand_conv(inputs, if_act=True)
        y = self._bottleneck_conv(y, if_act=True)
        y = self._linear_conv(y, if_act=False)
        if ifshortcut:
            y = fluid.layers.elementwise_add(inputs, y)
        return y


class InvresiBlocks(fluid.dygraph.Layer):
    def __init__(self, in_c, t, c, n, s, name=None):
        super(InvresiBlocks, self).__init__(name)

        self._first_block = InvertedResidualUnit(
            name=name + "_1",
            num_in_filter=in_c,
            num_filters=c,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t)

        self._inv_blocks = []
        for i in range(1, n):
            tmp = self.add_sublayer(
                sublayer=InvertedResidualUnit(
                    name=name + "_" + str(i + 1),
                    num_in_filter=c,
                    num_filters=c,
                    stride=1,
                    filter_size=3,
                    padding=1,
                    expansion_factor=t),
                name=name + "_" + str(i + 1))
            self._inv_blocks.append(tmp)

    def forward(self, inputs):
        y = self._first_block(inputs, ifshortcut=False)
        for inv_block in self._inv_blocks:
            y = inv_block(y, ifshortcut=True)
        return y


class MobileNetV2(fluid.dygraph.Layer):
    def __init__(self, name, scale=1.0):
        super(MobileNetV2, self).__init__(name)
        self.scale = scale

        bottleneck_params_list = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        #1. conv1 
        self._conv1 = ConvBNLayer(
            name="conv1_1",
            num_filters=int(32 * scale),
            filter_size=3,
            stride=2,
            padding=1)

        #2. bottleneck sequences
        self._invl = []
        i = 1
        in_c = int(32 * scale)
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            tmp = self.add_sublayer(
                sublayer=InvresiBlocks(
                    name='conv' + str(i),
                    in_c=in_c,
                    t=t,
                    c=int(c * scale),
                    n=n,
                    s=s),
                name='conv' + str(i))
            self._invl.append(tmp)
            in_c = int(c * scale)

        #3. last_conv
        self._conv9 = ConvBNLayer(
            name="conv9",
            num_filters=int(1280 * scale) if scale > 1.0 else 1280,
            filter_size=1,
            stride=1,
            padding=0)

        #4. pool
        self._pool2d_avg = Pool2D(
            name_scope="pool", pool_type='avg', global_pooling=True)

        #5. fc
        tmp_param = ParamAttr(name="fc10_weights")
        self._fc = FC(name_scope="fc",
                      size=args.class_dim,
                      param_attr=tmp_param,
                      bias_attr=ParamAttr(name="fc10_offset"))

    def forward(self, inputs):
        y = self._conv1(inputs, if_act=True)
        for inv in self._invl:
            y = inv(y)
        y = self._conv9(y, if_act=True)
        y = self._pool2d_avg(y)
        y = self._fc(y)
        return y


def eval(net, test_data_loader, eop):
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0
    t_last = 0
    for img, label in test_data_loader():
        t1 = time.time()
        label = to_variable(label.numpy().astype('int64').reshape(
            int(args.batch_size / paddle.fluid.core.get_cuda_device_count()),
            1))
        out = net(img)
        softmax_out = fluid.layers.softmax(out, use_cudnn=False)
        loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
        avg_loss = fluid.layers.mean(x=loss)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
        t2 = time.time()
        print( "test | epoch id: %d, avg_loss %0.5f acc_top1 %0.5f acc_top5 %0.5f %2.4f sec read_t:%2.4f" % \
                (eop, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy(), t2 - t1 , t1 - t_last))
        sys.stdout.flush()
        total_loss += avg_loss.numpy()
        total_acc1 += acc_top1.numpy()
        total_acc5 += acc_top5.numpy()
        total_sample += 1
        t_last = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("final eval loss %0.3f acc1 %0.3f acc5 %0.3f" % \
          (total_loss / total_sample, \
           total_acc1 / total_sample, total_acc5 / total_sample))
    sys.stdout.flush()


def train_mobilenet_v2():
    epoch = args.num_epochs
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
        net = MobileNetV2("mobilenet_v2", scale=c_scale)
        optimizer = create_optimizer(args)
        if args.use_data_parallel:
            net = fluid.dygraph.parallel.DataParallel(net, strategy)
        train_data_loader, train_data = utility.create_data_loader(
            is_train=True, args=args)
        test_data_loader, test_data = utility.create_data_loader(
            is_train=False, args=args)
        num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
        imagenet_reader = reader.ImageNetReader(
            0)  #(0 if num_trainers > 1 else None)
        train_reader = imagenet_reader.train(settings=args)
        test_reader = imagenet_reader.val(settings=args)
        train_data_loader.set_sample_list_generator(train_reader, place)
        test_data_loader.set_sample_list_generator(test_reader, place)
        for eop in range(epoch):
            if num_trainers > 1:
                imagenet_reader.set_shuffle_seed(eop + (
                    args.random_seed if args.random_seed else 0))
            net.train()
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0
            batch_id = 0
            t_last = 0
            for img, label in train_data_loader():
                t1 = time.time()
                label = to_variable(label.numpy().astype('int64').reshape(
                    int(args.batch_size /
                        paddle.fluid.core.get_cuda_device_count()), 1))
                t_start = time.time()
                out = net(img)
                t_end = time.time()
                softmax_out = fluid.layers.softmax(out, use_cudnn=False)
                loss = fluid.layers.cross_entropy(
                    input=softmax_out, label=label)
                avg_loss = fluid.layers.mean(x=loss)
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
                t_start_back = time.time()
                if args.use_data_parallel:
                    avg_loss = net.scale_loss(avg_loss)
                    avg_loss.backward()
                    net.apply_collective_grads()
                else:
                    avg_loss.backward()
                t_end_back = time.time()
                optimizer.minimize(avg_loss)
                net.clear_gradients()
                t2 = time.time()
                train_batch_elapse = t2 - t1
                if batch_id % args.print_step == 0:
                    print( "epoch id: %d, batch step: %d,  avg_loss %0.5f acc_top1 %0.5f acc_top5 %0.5f %2.4f sec net_t:%2.4f back_t:%2.4f read_t:%2.4f" % \
                            (eop, batch_id, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy(), train_batch_elapse,
                              t_end - t_start, t_end_back - t_start_back,  t1 - t_last))
                    sys.stdout.flush()
                total_loss += avg_loss.numpy()
                total_acc1 += acc_top1.numpy()
                total_acc5 += acc_top5.numpy()
                total_sample += 1
                batch_id += 1
                t_last = time.time()
            if args.ce:
                print("kpis\ttrain_acc1\t%0.3f" % (total_acc1 / total_sample))
                print("kpis\ttrain_acc5\t%0.3f" % (total_acc5 / total_sample))
                print("kpis\ttrain_loss\t%0.3f" % (total_loss / total_sample))
            print("epoch %d | batch step %d, loss %0.3f acc1 %0.3f acc5 %0.3f %2.4f sec" % \
                  (eop, batch_id, total_loss / total_sample, \
                   total_acc1 / total_sample, total_acc5 / total_sample, train_batch_elapse))
            net.eval()
            eval(net, test_data_loader, eop)
            save_parameters = (not args.use_data_parallel) or (
                args.use_data_parallel and
                fluid.dygraph.parallel.Env().local_rank == 0)
            if save_parameters:
                fluid.save_dygraph(net.state_dict(), 'mobilenet_v2_params')


if __name__ == '__main__':
    train_mobilenet_v2()
