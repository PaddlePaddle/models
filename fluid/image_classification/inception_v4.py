#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import numpy
import paddle.v2 as paddle
import paddle.fluid as fluid
import math

__all__ = ['inception_v4']


def img_conv(input, num_filters, filter_size, stride, padding, act='relu'):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None)
    norm = fluid.layers.batch_norm(input=conv, act=act)
    return norm


def stem(input):
    conv0 = img_conv(
        input=input, num_filters=32, filter_size=3, stride=2, padding=1)
    conv1 = img_conv(
        input=conv0, num_filters=32, filter_size=3, stride=1, padding=1)
    conv2 = img_conv(
        input=conv1, num_filters=64, filter_size=3, stride=1, padding=1)

    def block0(input):
        pool0 = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=2,
            # add
            pool_padding=1,
            pool_type='max')
        conv0 = img_conv(
            input=input, num_filters=96, filter_size=3, stride=2, padding=1)
        return fluid.layers.concat(input=[pool0, conv0], axis=1)

    def block1(input):
        l_conv0 = img_conv(
            input=input, num_filters=64, filter_size=1, stride=1, padding=0)
        l_conv1 = img_conv(
            input=l_conv0, num_filters=96, filter_size=3, stride=1, padding=1)
        r_conv0 = img_conv(
            input=input, num_filters=64, filter_size=1, stride=1, padding=0)
        r_conv1 = img_conv(
            input=r_conv0,
            num_filters=64,
            filter_size=(7, 1),
            stride=1,
            padding=(3, 0))
        r_conv2 = img_conv(
            input=r_conv1,
            num_filters=64,
            filter_size=(1, 7),
            stride=1,
            padding=(0, 3))
        r_conv3 = img_conv(
            input=r_conv2, num_filters=96, filter_size=3, stride=1, padding=1)
        return fluid.layers.concat(input=[l_conv1, r_conv3], axis=1)

    def block2(input):
        conv0 = img_conv(
            input=input, num_filters=192, filter_size=3, stride=2, padding=1)
        pool0 = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=2,
            # new add
            pool_padding=1,
            pool_type='max')
        return fluid.layers.concat(input=[conv0, pool0], axis=1)

    conv3 = block0(conv2)
    conv4 = block1(conv3)
    conv5 = block2(conv4)
    return conv5


def Inception_A(input, depth):
    b0_pool0 = fluid.layers.pool2d(
        input=input,
        pool_size=3,
        pool_stride=1,
        pool_padding=1,
        pool_type='avg')
    b0_conv0 = img_conv(
        input=b0_pool0, num_filters=96, filter_size=1, stride=1, padding=0)
    b1_conv0 = img_conv(
        input=input, num_filters=96, filter_size=1, stride=1, padding=0)
    b2_conv0 = img_conv(
        input=input, num_filters=64, filter_size=1, stride=1, padding=0)
    b2_conv1 = img_conv(
        input=b2_conv0, num_filters=96, filter_size=3, stride=1, padding=1)
    b3_conv0 = img_conv(
        input=input, num_filters=64, filter_size=1, stride=1, padding=0)
    b3_conv1 = img_conv(
        input=b3_conv0, num_filters=96, filter_size=3, stride=1, padding=1)
    b3_conv2 = img_conv(
        input=b3_conv1, num_filters=96, filter_size=3, stride=1, padding=1)
    return fluid.layers.concat(
        input=[b0_conv0, b1_conv0, b2_conv1, b3_conv2], axis=1)


def Inception_B(input, depth):
    b0_pool0 = fluid.layers.pool2d(
        input=input,
        pool_size=3,
        pool_stride=1,
        pool_padding=1,
        pool_type='avg')
    b0_conv0 = img_conv(
        input=b0_pool0, num_filters=128, filter_size=1, stride=1, padding=0)
    b1_conv0 = img_conv(
        input=input, num_filters=384, filter_size=1, stride=1, padding=0)
    b2_conv0 = img_conv(
        input=input, num_filters=192, filter_size=1, stride=1, padding=0)
    b2_conv1 = img_conv(
        input=b2_conv0,
        num_filters=224,
        filter_size=(1, 7),
        stride=1,
        padding=(0, 3))
    b2_conv2 = img_conv(
        input=b2_conv1,
        num_filters=256,
        filter_size=(7, 1),
        stride=1,
        padding=(3, 0))
    b3_conv0 = img_conv(
        input=input, num_filters=192, filter_size=1, stride=1, padding=0)
    b3_conv1 = img_conv(
        input=b3_conv0,
        num_filters=192,
        filter_size=(1, 7),
        stride=1,
        padding=(0, 3))
    b3_conv2 = img_conv(
        input=b3_conv1,
        num_filters=224,
        filter_size=(7, 1),
        stride=1,
        padding=(3, 0))
    b3_conv3 = img_conv(
        input=b3_conv2,
        num_filters=224,
        filter_size=(1, 7),
        stride=1,
        padding=(0, 3))
    b3_conv4 = img_conv(
        input=b3_conv3,
        num_filters=256,
        filter_size=(7, 1),
        stride=1,
        padding=(3, 0))
    return fluid.layers.concat(
        input=[b0_conv0, b1_conv0, b2_conv2, b3_conv4], axis=1)


def Inception_C(input, depth):
    b0_pool0 = fluid.layers.pool2d(
        input=input,
        pool_size=3,
        pool_stride=1,
        pool_padding=1,
        pool_type='avg')
    b0_conv0 = img_conv(
        input=b0_pool0, num_filters=256, filter_size=1, stride=1, padding=0)
    b1_conv0 = img_conv(
        input=input, num_filters=256, filter_size=1, stride=1, padding=0)
    b2_conv0 = img_conv(
        input=input, num_filters=384, filter_size=1, stride=1, padding=0)
    b2_conv1 = img_conv(
        input=b2_conv0,
        num_filters=256,
        filter_size=(1, 3),
        stride=1,
        padding=(0, 1))
    b2_conv2 = img_conv(
        input=b2_conv0,
        num_filters=256,
        filter_size=(3, 1),
        stride=1,
        padding=(1, 0))
    b3_conv0 = img_conv(
        input=input, num_filters=384, filter_size=1, stride=1, padding=0)
    b3_conv1 = img_conv(
        input=b3_conv0,
        num_filters=448,
        filter_size=(1, 3),
        stride=1,
        padding=(0, 1))
    b3_conv2 = img_conv(
        input=b3_conv1,
        num_filters=512,
        filter_size=(3, 1),
        stride=1,
        padding=(1, 0))
    b3_conv3 = img_conv(
        input=b3_conv2,
        num_filters=256,
        filter_size=(3, 1),
        stride=1,
        padding=(1, 0))
    b3_conv4 = img_conv(
        input=b3_conv2,
        num_filters=256,
        filter_size=(1, 3),
        stride=1,
        padding=(0, 1))
    return fluid.layers.concat(
        input=[b0_conv0, b1_conv0, b2_conv1, b2_conv2, b3_conv3, b3_conv4],
        axis=1)


def Reduction_A(input):
    b0_pool0 = fluid.layers.pool2d(
        input=input,
        pool_size=3,
        pool_stride=2,
        #new add
        pool_padding=1,
        pool_type='max')
    b1_conv0 = img_conv(
        input=input, num_filters=384, filter_size=3, stride=2, padding=1)
    b2_conv0 = img_conv(
        input=input, num_filters=192, filter_size=1, stride=1, padding=0)
    b2_conv1 = img_conv(
        input=b2_conv0, num_filters=224, filter_size=3, stride=1, padding=1)
    b2_conv2 = img_conv(
        input=b2_conv1, num_filters=256, filter_size=3, stride=2, padding=1)
    return fluid.layers.concat(input=[b0_pool0, b1_conv0, b2_conv2], axis=1)


def Reduction_B(input):
    b0_pool0 = fluid.layers.pool2d(
        input=input,
        pool_size=3,
        pool_stride=2,
        #new add
        pool_padding=1,
        pool_type='max')
    b1_conv0 = img_conv(
        input=input, num_filters=192, filter_size=1, stride=1, padding=0)
    b1_conv1 = img_conv(
        input=b1_conv0, num_filters=192, filter_size=3, stride=2, padding=1)
    b2_conv0 = img_conv(
        input=input, num_filters=256, filter_size=1, stride=1, padding=0)
    b2_conv1 = img_conv(
        input=b2_conv0,
        num_filters=256,
        filter_size=(1, 7),
        stride=1,
        padding=(0, 3))
    b2_conv2 = img_conv(
        input=b2_conv1,
        num_filters=320,
        filter_size=(7, 1),
        stride=1,
        padding=(3, 0))
    b2_conv3 = img_conv(
        input=b2_conv2, num_filters=320, filter_size=3, stride=2, padding=1)
    return fluid.layers.concat(input=[b0_pool0, b1_conv1, b2_conv3], axis=1)


def inception_v4(input, class_dim):
    conv = stem(input)

    for i in range(4):
        conv = Inception_A(conv, i)
    conv = Reduction_A(conv)
    for i in range(7):
        conv = Inception_B(conv, i)
    conv = Reduction_B(conv)
    for i in range(3):
        conv = Inception_C(conv, i)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=7, pool_stride=1, pool_type='avg')
    drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
    return out


def train(use_cuda,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir='model'):

    class_dim = 102
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net = inception_v4(image, class_dim)

    predict = fluid.layers.fc(input=net, size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=predict, label=label)

    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.flowers.train(), buf_size=1000),
        batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.flowers.valid(), batch_size=batch_size)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program([avg_cost, acc])

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    exe.run(fluid.default_startup_program())

    loss = 0.0
    for pass_id in range(num_passes):
        for batch_id, data in enumerate(train_reader()):
            exe.run(feed=feeder.feed(data))

            if (batch_id % 10) == 0:
                acc_list = []
                avg_loss_list = []
                for tid, test_data in enumerate(test_reader()):
                    loss_t, acc_t = exe.run(program=inference_program,
                                            feed=feeder.feed(test_data),
                                            fetch_list=[avg_cost, acc])
                    if math.isnan(float(loss_t)):
                        sys.exit("got NaN loss, training failed.")
                    acc_list.append(float(acc_t))
                    avg_loss_list.append(float(loss_t))
                    break  # Use 1 segment for speeding up CI

                acc_value = numpy.array(acc_list).mean()
                avg_loss_value = numpy.array(avg_loss_list).mean()

                print(
                    'PassID {0:1}, BatchID {1:04}, Test Loss {2:2.2}, Acc {3:2.2}'.
                    format(pass_id, batch_id + 1,
                           float(avg_loss_value), float(acc_value)))

                if acc_value > 0.01:  # Low threshold for speeding up CI
                    fluid.io.save_inference_model(model_save_dir, ["image"],
                                                  [predict], exe)
                    return


if __name__ == '__main__':
    train(use_cuda=False, learning_rate=0.001, batch_size=128, num_passes=1)
