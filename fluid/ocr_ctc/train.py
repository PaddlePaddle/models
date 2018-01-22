#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import print_function

import sys

import paddle.v2 as paddle
import paddle.v2.fluid as fluid


def ocr_conv(input, num, with_bn):
    assert (num % 4 == 0)

    def conv_block(input, filter_size, group_size, with_bn):
        return fluid.nets.img_conv_group(
            input=input,
            conv_num_filter=[num_filter] * groups,
            pool_size=2,
            pool_stride=2,
            conv_padding=1,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=with_bn,
            pool_type='max')

    conv1 = conv_block(input, 16, (num / 4), with_bn)
    conv2 = conv_block(conv1, 32, (num / 4), with_bn)
    conv3 = conv_block(conv2, 64, (num / 4), with_bn)
    conv4 = conv_block(conv3, 128, (num / 4), with_bn)
    return conv4


num_classes = 9054
data_shape = [3, 32, 32]

images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# encoder part
conv_features = ocr_convs(imges, 8, True)

sliced_feature = fluid.layers.im2sequence(
    input=conv_features,
    stride_x=1,
    stride_y=1,
    block_x=1,
    block_y=3, )

gru_forward = fluid.layers.gru(input=sliced_feature, size=200, act="relu")
gru_backward = fluid.layers.gru(input=sliced_feature,
                                size=200,
                                reverse=True,
                                act="relu")

out = fluid.layers.fc(input=[gru_forward, gru_backward], size=num_classes + 1)
cost = fluid.layers.warpctc(
    input=out,
    label=label,
    size=num_classes + 1,
    blank=num_classes,
    norm_by_times=True)

optimizer = fluid.optimizer.Momentum(
    learning_rate=((1.0e-3) / 16), momentum=0.9)
opts = optimizer.minimize(cost)

decoded_out = fluid.layers.ctc_greedy_decoder(input=output, blank=class_num)
error = fluid.evaluator.EditDistance(input=decoded_out, label=label)

BATCH_SIZE = 16
PASS_NUM = 1

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.cifar.train10(), buf_size=128 * 10),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
exe.run(fluid.default_startup_program())

for pass_id in range(PASS_NUM):
    accuracy.reset(exe)
    for data in train_reader():
        loss, acc = exe.run(fluid.default_main_program(),
                            feed=feeder.feed(data),
                            fetch_list=[avg_cost] + accuracy.metrics)
        pass_acc = accuracy.eval(exe)
        print("loss:" + str(loss) + " acc:" + str(acc) + " pass_acc:" + str(
            pass_acc))
        # this model is slow, so if we can train two mini batch, we think it works properly.
        exit(0)
exit(1)
