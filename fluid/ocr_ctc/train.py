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
import sys
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
from paddle.v2.fluid import core
import numpy as np
import dummy_reader


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int32")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = core.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def ocr_conv(input, num, with_bn, param_attrs):
    assert (num % 4 == 0)

    def conv_block(input, filter_size, group_size, with_bn):
        return fluid.nets.img_conv_group(
            input=input,
            conv_num_filter=[filter_size] * group_size,
            pool_size=2,
            pool_stride=2,
            conv_padding=1,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=with_bn,
            pool_type='max',
            param_attr=param_attrs)

    conv1 = conv_block(input, 16, (num / 4), with_bn)
    conv2 = conv_block(conv1, 32, (num / 4), with_bn)
    conv3 = conv_block(conv2, 64, (num / 4), with_bn)
    conv4 = conv_block(conv3, 128, (num / 4), with_bn)
    return conv4


def ocr_ctc_net(images, num_classes, param_attrs):
    conv_features = ocr_conv(images, 8, True, param_attrs)
    sliced_feature = fluid.layers.im2sequence(
        input=conv_features, stride=[1, 1], filter_size=[1, 3])
    gru_forward = fluid.layers.dynamic_gru(
        input=sliced_feature, size=128, param_attr=param_attrs)
    gru_backward = fluid.layers.dynamic_gru(
        input=sliced_feature, size=128, is_reverse=True, param_attr=param_attrs)

    fc_out = fluid.layers.fc(input=[gru_forward, gru_backward],
                             size=num_classes + 1,
                             param_attr=param_attrs)
    return fc_out


def get_feeder_data(data, place):
    pixel_tensor = core.LoDTensor()
    pixel_data = np.concatenate(
        map(lambda x: x[0][np.newaxis, :], data), axis=0).astype("float32")
    pixel_tensor.set(pixel_data, place)
    label_tensor = to_lodtensor(map(lambda x: x[1], data), place)
    return {"pixel": pixel_tensor, "label": label_tensor}


def train(num_classes=20,
          l2=0.0005 * 16,
          clip_threshold=10,
          data_reader=dummy_reader,
          learning_rate=((1.0e-3) / 16),
          momentum=0.9,
          batch_size=4,
          pass_num=2):

    param_attrs = fluid.ParamAttr(
        regularizer=fluid.regularizer.L2Decay(l2),
        gradient_clip=fluid.clip.GradientClipByValue(clip_threshold))
    data_shape = data_reader.data_shape()
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[1], dtype='int32', lod_level=1)

    fc_out = ocr_ctc_net(images, num_classes, param_attrs)

    cost = fluid.layers.warpctc(
        input=fc_out,
        label=label,
        size=num_classes + 1,
        blank=num_classes,
        norm_by_times=True)
    avg_cost = fluid.layers.mean(x=cost)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate, momentum=momentum)
    opts = optimizer.minimize(cost)

    decoded_out = fluid.layers.ctc_greedy_decoder(
        input=fc_out, blank=num_classes)
    casted_label = fluid.layers.cast(x=label, dtype='int64')
    error_evaluator = fluid.evaluator.EditDistance(
        input=decoded_out, label=casted_label)

    train_reader = paddle.batch(data_reader.train(), batch_size=batch_size)
    test_reader = paddle.batch(data_reader.test(), batch_size=batch_size)

    #place = fluid.CPUPlace()
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
    exe.run(fluid.default_startup_program())

    inference_program = fluid.io.get_inference_program(error_evaluator)
    for pass_id in range(pass_num):
        error_evaluator.reset(exe)
        batch_id = 0
        for data in train_reader():
            loss, batch_edit_distance, _, _ = exe.run(
                fluid.default_main_program(),
                feed=get_feeder_data(data, place),
                fetch_list=[avg_cost] + error_evaluator.metrics)
            print "Pass[%d], batch[%d]; loss: %s; edit distance: %s" % (
                pass_id, batch_id, loss[0], batch_edit_distance[0])
            batch_id += 1

        train_edit_distance = error_evaluator.eval(exe)
        print "End pass[%d]; train data edit_distance: %s" % (
            pass_id, str(train_edit_distance))

        # test
        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        test_edit_distance = error_evaluator.eval(exe)
        print "End pass[%d]; test data edit_distance: %s" % (
            pass_id, str(test_edit_distance))


if __name__ == "__main__":
    train()
