#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.v2 as paddle
import paddle.fluid as fluid
import numpy
import sys
from functools import partial

import math
import os

EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 100


use_cuda = False # set to True if training with GPU

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)


def inference_program(is_sparse):
    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    fourth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')

    embed_first = fluid.layers.embedding(
        input=first_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.layers.embedding(
        input=second_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.layers.embedding(
        input=third_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_fourth = fluid.layers.embedding(
        input=fourth_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    concat_embed = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
    hidden1 = fluid.layers.fc(
        input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
    return predict_word


def train_program(is_sparse):
    # The declaration of 'next_word' must be after the invoking of inference_program,
    # or the data input order of train program would be [next_word, firstw, secondw,
    # thirdw, fourthw], which is not correct.
    predict_word = inference_program(is_sparse)
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.AdagradOptimizer(
        learning_rate=3e-3,
        regularization=fluid.regularizer.L2DecayRegularizer(8e-4))


def train(use_cuda, train_program, params_dirname):
    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            outs = trainer.test(
                reader=test_reader,
                feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])
            avg_cost = outs[0]

            if event.step % 10 == 0:
                print "Step %d: Average Cost %f" % (event.step, avg_cost)

            # If average cost is lower than 5.8, we consider the model good enough to stop.
            # Note 5.8 is a relatively high value. In order to get a better model, one should
            # aim for avg_cost lower than 3.5. But the training could take longer time.
            if avg_cost < 5.8:
                trainer.save_params(params_dirname)
                trainer.stop()

            if math.isnan(avg_cost):
                sys.exit("got NaN loss, training failed.")

    trainer = fluid.Trainer(
        train_func=train_program,
        # optimizer=fluid.optimizer.SGD(learning_rate=0.001),
        optimizer_func=optimizer_func,
        place=place)

    trainer.train(
        reader=train_reader,
        num_epochs=1,
        event_handler=event_handler,
        feed_order=['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw'])


def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)

    # Setup inputs by creating 4 LoDTensors representing 4 words. Here each word
    # is simply an index to look up for the corresponding word vector and hence
    # the shape of word (base_shape) should be [1]. The length-based level of
    # detail (lod) info of each LoDtensor should be [[1]] meaning there is only
    # one lod_level and there is only one sequence of one word on this level.
    # Note that lod info should be a list of lists.

    data1 = [[211]]  # 'among'
    data2 = [[6]]  # 'a'
    data3 = [[96]]  # 'group'
    data4 = [[4]]  # 'of'
    lod = [[1]]

    first_word = fluid.create_lod_tensor(data1, lod, place)
    second_word = fluid.create_lod_tensor(data2, lod, place)
    third_word = fluid.create_lod_tensor(data3, lod, place)
    fourth_word = fluid.create_lod_tensor(data4, lod, place)

    result = inferencer.infer(
        {
            'firstw': first_word,
            'secondw': second_word,
            'thirdw': third_word,
            'fourthw': fourth_word
        },
        return_numpy=False)

    print(numpy.array(result[0]))
    most_possible_word_index = numpy.argmax(result[0])
    print(most_possible_word_index)
    print([
        key for key, value in word_dict.iteritems()
        if value == most_possible_word_index
    ][0])


def main(use_cuda, is_sparse):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    params_dirname = "word2vec.inference.model"

    train(
        use_cuda=use_cuda,
        train_program=partial(train_program, is_sparse),
        params_dirname=params_dirname)

    infer(
        use_cuda=use_cuda,
        inference_program=partial(inference_program, is_sparse),
        params_dirname=params_dirname)


if __name__ == '__main__':
    main(use_cuda=use_cuda, is_sparse=True)