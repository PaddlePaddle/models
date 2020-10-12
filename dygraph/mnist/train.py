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
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')

    args = parser.parse_args()
    return args


class SimpleImgConvPool(paddle.nn.Layer):
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

        self._conv2d = paddle.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            weight_attr=None,
            bias_attr=None)
        self._act = act

        self._pool2d = paddle.fluid.dygraph.nn.Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = getattr(paddle.nn.functional, self._act)(x) if self._act else x
        x = self._pool2d(x)
        return x


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = paddle.nn.Linear(
            in_features=self.pool_2_shape,
            out_features=10,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    loc=0.0, scale=scale)))

    def forward(self, inputs, label=None):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = paddle.fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        x = paddle.nn.functional.softmax(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(1, 28, 28)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__


def test_mnist(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        img, label = data
        label.stop_gradient = True
        prediction, acc = model(img, label)
        loss = paddle.fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


def inference_mnist():
    if not args.use_gpu:
        place = paddle.CPUPlace()
    elif not args.use_data_parallel:
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CUDAPlace(paddle.fluid.dygraph.parallel.Env().dev_id)

    paddle.disable_static(place)
    mnist_infer = MNIST()
        # load checkpoint
    model_dict, _ = paddle.fluid.load_dygraph("save_temp")
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

    results = mnist_infer(paddle.to_tensor(data=tensor_img, dtype=None, place=None, stop_gradient=True))
    lab = np.argsort(results.numpy())
    print("Inference result of image/infer_3.png is: %d" % lab[0][-1])
    paddle.enable_static()


def train_mnist(args):
    epoch_num = args.epoch
    BATCH_SIZE = 64

    if not args.use_gpu:
        place = paddle.CPUPlace()
    elif not args.use_data_parallel:
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CUDAPlace(paddle.fluid.dygraph.parallel.Env().dev_id)

    paddle.disable_static(place)
    if args.ce:
        print("ce mode")
        seed = 33
        np.random.seed(seed)
        paddle.static.default_startup_program().random_seed = seed
        paddle.static.default_main_program().random_seed = seed

    if args.use_data_parallel:
        strategy = paddle.fluid.dygraph.parallel.prepare_context()
    mnist = MNIST()
    adam = paddle.optimizer.Adam(
        learning_rate=0.001, parameters=mnist.parameters())
    if args.use_data_parallel:
        mnist = paddle.fluid.dygraph.parallel.DataParallel(mnist, strategy)

    train_reader = paddle.batch(
        reader_decorator(paddle.dataset.mnist.train()),
        batch_size=BATCH_SIZE,
        drop_last=True)
    if args.use_data_parallel:
        train_reader = paddle.fluid.contrib.reader.distributed_batch_reader(
            train_reader)

    test_reader = paddle.batch(
        reader_decorator(paddle.dataset.mnist.test()),
        batch_size=BATCH_SIZE,
        drop_last=True)

    train_loader = paddle.io.DataLoader.from_generator(capacity=10)
    train_loader.set_sample_list_generator(train_reader, places=place)

    test_loader = paddle.io.DataLoader.from_generator(capacity=10)
    test_loader.set_sample_list_generator(test_reader, places=place)

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True

            cost, acc = mnist(img, label)

            loss = paddle.fluid.layers.cross_entropy(cost, label)
            avg_loss = paddle.mean(loss)

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
        test_cost, test_acc = test_mnist(test_loader, mnist, BATCH_SIZE)
        mnist.train()
        if args.ce:
            print("kpis\ttest_acc\t%s" % test_acc)
            print("kpis\ttest_cost\t%s" % test_cost)
        print("Loss at epoch {} , Test avg_loss is: {}, acc is: {}".format(
            epoch, test_cost, test_acc))

    save_parameters = (not args.use_data_parallel) or (
        args.use_data_parallel and
        paddle.fluid.dygraph.parallel.Env().local_rank == 0)
    if save_parameters:
        paddle.fluid.save_dygraph(mnist.state_dict(), "save_temp")

        print("checkpoint saved")

        inference_mnist()
    paddle.enable_static()


if __name__ == '__main__':
    args = parse_args()
    train_mnist(args)
