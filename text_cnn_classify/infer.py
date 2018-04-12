import numpy as np
import sys
import os
import argparse
import time

import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

from config import TestConfig as conf


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

    print("Done Inferring.")


if __name__ == '__main__':
    args = parse_args()
    with profiler.profiler("GPU", 'total') as prof:
        main(args.dict_path)
