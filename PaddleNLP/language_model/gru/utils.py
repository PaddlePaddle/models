import sys
import time
import numpy as np

import paddle.fluid as fluid
import paddle


def to_lodtensor(data, place):
    """ convert to LODtensor """
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


def prepare_data(batch_size,
                 buffer_size=1000,
                 word_freq_threshold=0,
                 enable_ce=False):
    """ prepare the English Pann Treebank (PTB) data """
    vocab = paddle.dataset.imikolov.build_dict(word_freq_threshold)
    if enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.imikolov.train(
                vocab,
                buffer_size,
                data_type=paddle.dataset.imikolov.DataType.SEQ),
            batch_size)
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.imikolov.train(
                    vocab,
                    buffer_size,
                    data_type=paddle.dataset.imikolov.DataType.SEQ),
                buf_size=buffer_size),
            batch_size)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(
            vocab, buffer_size, data_type=paddle.dataset.imikolov.DataType.SEQ),
        batch_size)
    return vocab, train_reader, test_reader
