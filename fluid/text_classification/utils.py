import sys
import time
import numpy as np

import paddle.fluid as fluid
import paddle.v2 as paddle

import light_imdb
import tiny_imdb


def to_lodtensor(data, place):
    """
    convert to LODtensor
    """
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


def load_vocab(filename):
    """
    load imdb vocabulary
    """
    vocab = {}
    with open(filename) as f:
        wid = 0
        for line in f:
            vocab[line.strip()] = wid
            wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


def data2tensor(data, place):
    """
    data2tensor
    """
    input_seq = to_lodtensor(map(lambda x: x[0], data), place)
    y_data = np.array(map(lambda x: x[1], data)).astype("int64")
    y_data = y_data.reshape([-1, 1])
    return {"words": input_seq, "label": y_data}


def prepare_data(data_type="imdb",
                 self_dict=False,
                 batch_size=128,
                 buf_size=50000):
    """
    prepare data
    """
    if self_dict:
        word_dict = load_vocab(data_type + ".vocab")
    else:
        if data_type == "imdb":
            word_dict = paddle.dataset.imdb.word_dict()
        elif data_type == "light_imdb":
            word_dict = light_imdb.word_dict()
        elif data_type == "tiny_imdb":
            word_dict = tiny_imdb.word_dict()
        else:
            raise RuntimeError("No such dataset")

    if data_type == "imdb":
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.imdb.train(word_dict), buf_size=buf_size),
            batch_size=batch_size)

        test_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.imdb.test(word_dict), buf_size=buf_size),
            batch_size=batch_size)

    elif data_type == "light_imdb":
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                light_imdb.train(word_dict), buf_size=buf_size),
            batch_size=batch_size)

        test_reader = paddle.batch(
            paddle.reader.shuffle(
                light_imdb.test(word_dict), buf_size=buf_size),
            batch_size=batch_size)

    elif data_type == "tiny_imdb":
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                tiny_imdb.train(word_dict), buf_size=buf_size),
            batch_size=batch_size)

        test_reader = paddle.batch(
            paddle.reader.shuffle(
                tiny_imdb.test(word_dict), buf_size=buf_size),
            batch_size=batch_size)
    else:
        raise RuntimeError("no such dataset")

    return word_dict, train_reader, test_reader
