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

import os
import sys
import time
import multiprocessing

import paddle
import paddle.fluid as fluid


def train(network, dict_dim, lr, save_dirname, training_data_dirname, pass_num,
          thread_num, batch_size):
    file_names = os.listdir(training_data_dirname)
    filelist = []
    for i in range(0, len(file_names)):
        if file_names[i] == 'data_feed.proto':
            continue
        filelist.append(os.path.join(training_data_dirname, file_names[i]))

    dataset = fluid.DataFeedDesc(
        os.path.join(training_data_dirname, 'data_feed.proto'))
    dataset.set_batch_size(
        batch_size)  # datafeed should be assigned a batch size
    dataset.set_use_slots(['words', 'label'])

    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    avg_cost, acc, prediction = network(data, label, dict_dim)
    optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    opt_ops, weight_and_grad = optimizer.minimize(avg_cost)

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    place = fluid.CPUPlace()
    executor = fluid.Executor(place)
    executor.run(startup_program)

    async_executor = fluid.AsyncExecutor(place)
    for i in range(pass_num):
        pass_start = time.time()
        async_executor.run(main_program,
                           dataset,
                           filelist,
                           thread_num, [acc],
                           debug=False)
        print('pass_id: %u pass_time_cost %f' % (i, time.time() - pass_start))
        fluid.io.save_inference_model('%s/epoch%d.model' % (save_dirname, i),
                                      [data.name, label.name], [acc], executor)


if __name__ == "__main__":
    if __package__ is None:
        from os import sys, path
        sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from nets import bow_net, cnn_net, lstm_net, gru_net
    from utils import load_vocab

    batch_size = 4
    lr = 0.002
    pass_num = 30
    save_dirname = ""
    thread_num = multiprocessing.cpu_count()

    if sys.argv[1] == "bow":
        network = bow_net
        batch_size = 128
        save_dirname = "bow_model"
    elif sys.argv[1] == "cnn":
        network = cnn_net
        lr = 0.01
        save_dirname = "cnn_model"
    elif sys.argv[1] == "lstm":
        network = lstm_net
        lr = 0.05
        save_dirname = "lstm_model"
    elif sys.argv[1] == "gru":
        network = gru_net
        batch_size = 128
        lr = 0.05
        save_dirname = "gru_model"

    training_data_dirname = 'train_data/'
    if len(sys.argv) == 3:
        training_data_dirname = sys.argv[2]

    if len(sys.argv) == 4:
        if thread_num >= int(sys.argv[3]):
            thread_num = int(sys.argv[3])

    vocab = load_vocab('imdb.vocab')
    dict_dim = len(vocab)

    train(network, dict_dim, lr, save_dirname, training_data_dirname, pass_num,
          thread_num, batch_size)
