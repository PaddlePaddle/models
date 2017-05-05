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
import paddle.v2 as paddle
import gzip


def convolution_net(input_dim, class_dim=2, emb_dim=128, hid_dim=128):
    # input layers
    data = paddle.layer.data("word",
                             paddle.data_type.integer_value_sequence(input_dim))
    lbl = paddle.layer.data("label", paddle.data_type.integer_value(2))

    #embedding layer
    emb = paddle.layer.embedding(input=data, size=emb_dim)

    # convolution layers with max pooling
    conv_3 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=3, hidden_size=hid_dim)
    conv_4 = paddle.networks.sequence_conv_pool(
        input=emb, context_len=4, hidden_size=hid_dim)

    # fc and output layer
    output = paddle.layer.fc(
        input=[conv_3, conv_4], size=class_dim, act=paddle.activation.Softmax())

    cost = paddle.layer.classification_cost(input=output, label=lbl)

    return cost, output


def train_cnn_model(num_pass):
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
    [cost, _] = convolution_net(dict_dim, class_dim=class_dim)
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
            with gzip.open("cnn_params.tar.gz", 'w') as f:
                parameters.to_tar(f)

    # begin training network
    feeding = {'word': 0, 'label': 1}
    trainer.train(
        reader=train_reader,
        event_handler=event_handler,
        feeding=feeding,
        num_passes=num_pass)

    print("Training finished.")


def cnn_infer():
    print("Begin to predict...")

    word_dict = paddle.dataset.imdb.word_dict()
    dict_dim = len(word_dict)
    class_dim = 2

    [_, output] = convolution_net(dict_dim, class_dim=class_dim)
    parameters = paddle.parameters.Parameters.from_tar(
        gzip.open("cnn_params.tar.gz"))

    infer_data = []
    infer_label_data = []
    infer_data_num = 100
    for item in paddle.dataset.imdb.test(word_dict):
        infer_data.append([item[0]])
        infer_label_data.append(item[1])
        if len(infer_data) == infer_data_num:
            break

    predictions = paddle.infer(
        output_layer=output,
        parameters=parameters,
        input=infer_data,
        field=['value'])
    for i, prob in enumerate(predictions):
        print prob, infer_label_data[i]


if __name__ == "__main__":
    paddle.init(use_gpu=False, trainer_count=10)
    train_cnn_model(num_pass=10)
    cnn_infer()
