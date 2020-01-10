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

from __future__ import print_function
import argparse
import ast
import numpy as np
from PIL import Image
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.base import to_variable


def parse_args():
    parser = argparse.ArgumentParser("Training for Mnist.")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use data parallel mode to train the model."
    )
    parser.add_argument("-e", "--epoch", default=5, type=int, help="set epoch")
    parser.add_argument("--ce", action="store_true", help="run ce")
    args = parser.parse_args()
    return args


class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__()

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(self.pool_2_shape, 10,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.NormalInitializer(
                              loc=0.0, scale=scale)),
                      act="softmax")

    def forward(self, inputs, label=None):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


def test_mnist(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0].reshape(1, 28, 28)
                              for x in data]).astype('float32')
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)

        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = fluid.layers.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


def inference_mnist():
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        mnist_infer = MNIST()
        # load checkpoint
        model_dict, _ = fluid.load_dygraph("save_temp")
        mnist_infer.set_dict(model_dict)
        print("checkpoint loaded")

        # start evaluate mode
        mnist_infer.eval()

        def load_image(file):
            im = Image.open(file).convert('L')
            im = im.resize((28, 28), Image.ANTIALIAS)
            im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            return im

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tensor_img = load_image(cur_dir + '/image/infer_3.png')

        results = mnist_infer(to_variable(tensor_img))
        lab = np.argsort(results.numpy())
        print("Inference result of image/infer_3.png is: %d" % lab[0][-1])


def train_mnist(args):
    epoch_num = args.epoch
    BATCH_SIZE = 64

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
        mnist = MNIST()
        adam = AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
        if args.use_data_parallel:
            mnist = fluid.dygraph.parallel.DataParallel(mnist, strategy)

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                      for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                cost, acc = mnist(img, label)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                if args.use_data_parallel:
                    avg_loss = mnist.scale_loss(avg_loss)
                    avg_loss.backward()
                    mnist.apply_collective_grads()
                else:
                    avg_loss.backward()

                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print("Loss at epoch {} step {}: {:}".format(
                        epoch, batch_id, avg_loss.numpy()))

            mnist.eval()
            test_cost, test_acc = test_mnist(test_reader, mnist, BATCH_SIZE)
            mnist.train()
            if args.ce:
                print("kpis\ttest_acc\t%s" % test_acc)
                print("kpis\ttest_cost\t%s" % test_cost)
            print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
                epoch, test_cost, test_acc))

        save_parameters = (not args.use_data_parallel) or (
            args.use_data_parallel and
            fluid.dygraph.parallel.Env().local_rank == 0)
        if save_parameters:
            fluid.save_dygraph(mnist.state_dict(), "save_temp")
            
            print("checkpoint saved")

            inference_mnist()


if __name__ == '__main__':
    args = parse_args()
    train_mnist(args)
