# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import sys
import math
import paddle.v2 as paddle
import gzip


def fc_net(input_dim, class_dim=2, emb_dim=256):
    # input layers
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(input_dim))
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))

    # embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)
    # max pooling
    seq_pool = paddle.layer.pooling(
        input=emb, pooling_type=paddle.pooling.Max())

    # two hidden layers
    hd_layer_size = [128, 32]
    hd_layer_init_std = [1.0 / math.sqrt(s) / 3.0 for s in hd_layer_size]
    hd1 = paddle.layer.fc(
        input=seq_pool,
        size=hd_layer_size[0],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=hd_layer_init_std[0]))
    hd2 = paddle.layer.fc(
        input=hd1,
        size=hd_layer_size[1],
        act=paddle.activation.Tanh(),
        param_attr=paddle.attr.Param(initial_std=hd_layer_init_std[1]))

    # output layer
    output = paddle.layer.fc(
        input=hd2,
        size=class_dim,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(initial_std=1.0 / math.sqrt(class_dim) /
                                     3.0))

    cost = paddle.layer.classification_cost(input=output, label=lbl)

    return cost, output


def train_dnn_model(num_pass):
    # load word dictionary
    print 'load dictionary...'
    word_dict = paddle.dataset.imdb.word_dict()

    dict_dim = len(word_dict)
    class_dim = 2
    # define data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            lambda: paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=100)
    test_reader = paddle.batch(
        lambda: paddle.dataset.imdb.test(word_dict), batch_size=100)

    # network config
    [cost, _] = fc_net(dict_dim, class_dim=class_dim)
    # create parameters
    parameters = paddle.parameters.create(cost)
    # create optimizer
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=2e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    # create trainer
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=adam_optimizer)

    # Define end batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(reader=test_reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
            with gzip.open("dnn_params_pass" + str(event.pass_id) + ".tar.gz",
                           'w') as f:
                parameters.to_tar(f)

    # begin training network
    feeding = {'word': 0, 'label': 1}
    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_pass)

    print("Training finished.")


def dnn_infer(file_name):
    print("Begin to predict...")

    word_dict = paddle.dataset.imdb.word_dict()
    dict_dim = len(word_dict)
    class_dim = 2

    [_, output] = fc_net(dict_dim, class_dim=class_dim)
    parameters = paddle.parameters.Parameters.from_tar(gzip.open(file_name))

    infer_data = []
    infer_data_label = []
    for item in paddle.dataset.imdb.test(word_dict):
        infer_data.append([item[0]])
        infer_data_label.append(item[1])

    predictions = paddle.infer(
        output_layer=output,
        parameters=parameters,
        input=infer_data,
        field=['value'])
    for i, prob in enumerate(predictions):
        print prob, infer_data_label[i]


if __name__ == "__main__":
    paddle.init(use_gpu=False, trainer_count=4)
    num_pass = 5
    train_dnn_model(num_pass=num_pass)
    param_file_name = "dnn_params_pass" + str(num_pass - 1) + ".tar.gz"
    dnn_infer(file_name=param_file_name)
