import numpy as np
import sys
import os
import argparse
import time

import paddle.v2 as paddle
import paddle.v2.fluid as fluid

from config import TrainConfig as conf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dict_path',
        type=str,
        required=True,
        help="Path of the word dictionary.")
    return parser.parse_args()


# Define to_lodtensor function to process the sequential data.
def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


# Load the dictionary.
def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab


# Define the convolution model.
def conv_net(dict_dim,
             window_size=3,
             emb_dim=128,
             num_filters=128,
             fc0_dim=96,
             class_dim=2):

    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=num_filters,
        filter_size=window_size,
        act="tanh",
        pool_type="max")

    fc_0 = fluid.layers.fc(input=[conv_3], size=fc0_dim)

    prediction = fluid.layers.fc(input=[fc_0], size=class_dim, act="softmax")

    cost = fluid.layers.cross_entropy(input=prediction, label=label)

    avg_cost = fluid.layers.mean(x=cost)

    return data, label, prediction, avg_cost


def main(dict_path):
    word_dict = load_vocab(dict_path)
    word_dict["<unk>"] = len(word_dict)
    dict_dim = len(word_dict)
    print("The dictionary size is : %d" % dict_dim)

    data, label, prediction, avg_cost = conv_net(dict_dim)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=conf.learning_rate)
    sgd_optimizer.minimize(avg_cost)

    accuracy = fluid.evaluator.Accuracy(input=prediction, label=label)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        test_target = accuracy.metrics + accuracy.states
        inference_program = fluid.io.get_inference_program(test_target)

    # The training data set.
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=51200),
        batch_size=conf.batch_size)

    # The testing data set.
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.test(word_dict), buf_size=51200),
        batch_size=conf.batch_size)

    if conf.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    exe.run(fluid.default_startup_program())

    def test(exe):
        accuracy.reset(exe)
        for batch_id, data in enumerate(test_reader()):
            input_seq = to_lodtensor(map(lambda x: x[0], data), place)
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])
            acc = exe.run(inference_program,
                          feed={"words": input_seq,
                                "label": y_data})
        test_acc = accuracy.eval(exe)
        return test_acc

    total_time = 0.
    for pass_id in xrange(conf.num_passes):
        accuracy.reset(exe)
        start_time = time.time()
        for batch_id, data in enumerate(train_reader()):
            cost_val, acc_val = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost, accuracy.metrics[0]])
            pass_acc = accuracy.eval(exe)
            if batch_id and batch_id % conf.log_period == 0:
                print("Pass id: %d, batch id: %d, cost: %f, pass_acc %f" %
                      (pass_id, batch_id, cost_val, pass_acc))
        end_time = time.time()
        total_time += (end_time - start_time)
        pass_test_acc = test(exe)
        print("Pass id: %d, test_acc: %f" % (pass_id, pass_test_acc))
    print("Total train time: %f" % (total_time))


if __name__ == '__main__':
    args = parse_args()
    main(args.dict_path)
