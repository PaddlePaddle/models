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

import numpy as np
import argparse
import ast
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

from paddle.fluid import framework

import math
import sys
import time
import reader

IMAGENET1000 = 1281167
base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4


def parse_args():
    parser = argparse.ArgumentParser("Training for Resnet.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use data parallel mode to train the model."
    )
    parser.add_argument(
        "-e", "--epoch", default=120, type=int, help="set epoch")
    parser.add_argument(
        "-b", "--batch_size", default=32, type=int, help="set epoch")
    parser.add_argument("--ce", action="store_true", help="run ce")

    # NOTE:used in benchmark
    parser.add_argument(
        "--max_iter",
        default=0,
        type=int,
        help="the max iters to train, used in benchmark")
    parser.add_argument(
        "--class_dim",
        default=102,
        type=int,
        help="the class number of flowers dataset")
    parser.add_argument(
        "--use_imagenet_data",
        action="store_true",
        help="Use imagenet dataset instead of the flowers dataset(small dataset)"
    )
    parser.add_argument(
        '--data_dir',
        default="./data/ILSVRC2012",
        type=str,
        help="The ImageNet dataset root directory.")
    parser.add_argument(
        '--lower_scale',
        default=0.08,
        type=float,
        help="The value of lower_scale in ramdom_crop")
    parser.add_argument(
        '--lower_ratio',
        default=3. / 4.,
        type=float,
        help="The value of lower_ratio in ramdom_crop")
    parser.add_argument(
        '--upper_ratio',
        default=4. / 3.,
        type=float,
        help="The value of upper_ratio in ramdom_crop")
    parser.add_argument(
        '--resize_short_size',
        default=256,
        type=int,
        help="The value of resize_short_size")
    parser.add_argument(
        '--crop_size', default=224, type=int, help="The value of crop size")
    parser.add_argument(
        '--use_mixup', default=False, type=bool, help="Whether to use mixup")
    parser.add_argument(
        '--mixup_alpha',
        default=0.2,
        type=float,
        help="The value of mixup_alpha")
    parser.add_argument(
        '--reader_thread',
        default=8,
        type=int,
        help="The number of multi thread reader")
    parser.add_argument(
        '--reader_buf_size',
        default=16,
        type=int,
        help="The buf size of multi thread reader")
    parser.add_argument(
        '--interpolation',
        default=None,
        type=int,
        help="The interpolation mode")
    parser.add_argument(
        '--use_aa',
        default=False,
        type=bool,
        help="Whether to use auto augment")
    parser.add_argument(
        '--image_mean',
        nargs='+',
        type=float,
        default=[0.485, 0.456, 0.406],
        help="The mean of input image data")
    parser.add_argument(
        '--image_std',
        nargs='+',
        type=float,
        default=[0.229, 0.224, 0.225],
        help="The std of input image data")
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')

    args = parser.parse_args()
    return args


args = parse_args()
batch_size = args.batch_size


class TimeCostAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt = 0
        self.total_time = 0

    def record(self, usetime):
        self.cnt += 1
        self.total_time += usetime

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.total_time / self.cnt


def optimizer_setting(parameter_list=None):

    total_images = IMAGENET1000

    step = int(math.ceil(float(total_images) / batch_size))

    epochs = [30, 60, 90]
    bd = [step * e for e in epochs]

    lr = []
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    if fluid.in_dygraph_mode():
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay),
            parameter_list=parameter_list)
    else:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))

    return optimizer


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
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
    def __init__(self, layers=50, class_dim=102):
        super(ResNet, self).__init__()

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
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3, num_filters=64, filter_size=7, stride=2, act='relu')
        self.pool2d_max = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.bottleneck_block_list = []
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 4 * 1 * 1

        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(
            self.pool2d_avg_output,
            class_dim,
            act='softmax',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_output])
        y = self.out(y)
        return y


def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__


def eval(model, data):

    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0
    for batch_id, data in enumerate(data()):
        img = data[0]
        label = data[1]
        label.stop_gradient = True

        out = model(img)

        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

        #dy_out = avg_loss.numpy()

        #total_loss += dy_out
        total_acc1 += acc_top1.numpy()
        total_acc5 += acc_top5.numpy()
        total_sample += 1

        if batch_id % 10 == 0:
            print("test | batch step %d, acc1 %0.3f acc5 %0.3f" % \
                  ( batch_id, total_acc1 / total_sample, total_acc5 / total_sample))
    if args.ce:
        print("kpis\ttest_acc1\t%0.3f" % (total_acc1 / total_sample))
        print("kpis\ttest_acc5\t%0.3f" % (total_acc5 / total_sample))
    print("final eval acc1 %0.3f acc5 %0.3f" % \
          (total_acc1 / total_sample, total_acc5 / total_sample))


def train_resnet():
    epoch = args.epoch

    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    with fluid.dygraph.guard(place):
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        resnet = ResNet(class_dim=args.class_dim)
        optimizer = optimizer_setting(parameter_list=resnet.parameters())

        if args.use_data_parallel:
            resnet = fluid.dygraph.parallel.DataParallel(resnet, strategy)

        if args.use_imagenet_data:
            imagenet_reader = reader.ImageNetReader(0)
            train_reader = imagenet_reader.train(settings=args)
        else:
            train_reader = paddle.batch(
                reader_decorator(paddle.dataset.flowers.train(use_xmap=True)),
                batch_size=batch_size,
                drop_last=True)

        if args.use_imagenet_data:
            test_reader = imagenet_reader.val(settings=args)
        else:
            test_reader = paddle.batch(
                reader_decorator(paddle.dataset.flowers.test(use_xmap=True)),
                batch_size=batch_size,
                drop_last=True)

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=32,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=True)
        train_loader.set_sample_list_generator(train_reader, places=place)

        test_loader = fluid.io.DataLoader.from_generator(
            capacity=64,
            use_double_buffer=True,
            iterable=True,
            return_list=True,
            use_multiprocess=True)
        test_loader.set_sample_list_generator(test_reader, places=place)

        #NOTE: used in benchmark 
        total_batch_num = 0

        for eop in range(epoch):
            epoch_start = time.time()

            resnet.train()
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0

            train_batch_cost_avg = TimeCostAverage()
            train_reader_cost_avg = TimeCostAverage()
            batch_start = time.time()
            for batch_id, data in enumerate(train_loader()):
                #NOTE: used in benchmark
                if args.max_iter and total_batch_num == args.max_iter:
                    return

                train_reader_cost = time.time() - batch_start

                img, label = data
                label.stop_gradient = True

                out = resnet(img)
                loss = fluid.layers.cross_entropy(input=out, label=label)
                avg_loss = fluid.layers.mean(x=loss)

                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

                dy_out = avg_loss.numpy()

                if args.use_data_parallel:
                    avg_loss = resnet.scale_loss(avg_loss)
                    avg_loss.backward()
                    resnet.apply_collective_grads()
                else:
                    avg_loss.backward()

                optimizer.minimize(avg_loss)
                resnet.clear_gradients()

                total_loss += dy_out
                total_acc1 += acc_top1.numpy()
                total_acc5 += acc_top5.numpy()
                total_sample += 1

                train_batch_cost = time.time() - batch_start
                train_batch_cost_avg.record(train_batch_cost)
                train_reader_cost_avg.record(train_reader_cost)

                total_batch_num = total_batch_num + 1  #this is for benchmark
                if batch_id % 10 == 0:
                    ips = float(
                        args.batch_size) / train_batch_cost_avg.get_average()
                    print(
                        "[Epoch %d, batch %d] loss: %.5f, acc1: %.5f, acc5: %.5f, batch_cost: %.5f sec, reader_cost: %.5f sec, ips: %.5f images/sec"
                        % (eop, batch_id, total_loss / total_sample,
                           total_acc1 / total_sample, total_acc5 / total_sample,
                           train_batch_cost_avg.get_average(),
                           train_reader_cost_avg.get_average(), ips))
                    train_batch_cost_avg.reset()
                    train_reader_cost_avg.reset()
                batch_start = time.time()

            if args.ce:
                print("kpis\ttrain_acc1\t%0.3f" % (total_acc1 / total_sample))
                print("kpis\ttrain_acc5\t%0.3f" % (total_acc5 / total_sample))
                print("kpis\ttrain_loss\t%0.3f" % (total_loss / total_sample))

            train_epoch_cost = time.time() - epoch_start
            print(
                "[Epoch %d], loss %.5f, acc1 %.5f, acc5 %.5f, epoch_cost: %.5f s"
                % (eop, total_loss / total_sample, total_acc1 / total_sample,
                   total_acc5 / total_sample, train_epoch_cost))

            resnet.eval()
            eval(resnet, test_loader)

            save_parameters = (not args.use_data_parallel) or (
                args.use_data_parallel and
                fluid.dygraph.parallel.Env().local_rank == 0)
            if save_parameters:
                fluid.save_dygraph(resnet.state_dict(), 'resnet_params')


if __name__ == '__main__':
    train_resnet()
