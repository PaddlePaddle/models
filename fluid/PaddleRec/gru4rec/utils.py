import sys
import collections
import six
import time
import numpy as np
import paddle.fluid as fluid
import paddle
import os

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

def get_vocab_size(vocab_path):
    with open(vocab_path, "r") as rf:
        line = rf.readline()
        return int(line.strip())

def prepare_data(file_dir,
                 vocab_path,
                 batch_size,
                 buffer_size=1000,
                 word_freq_threshold=0,
                 is_train=True):
    """ prepare the English Pann Treebank (PTB) data """
    print("start constuct word dict")
    if is_train:
        vocab_size = get_vocab_size(vocab_path)
        reader = sort_batch(
            paddle.reader.shuffle(
                train(
                    file_dir, buffer_size, data_type=DataType.SEQ),
                buf_size=buffer_size),
            batch_size,
            batch_size * 20)
    else:
        reader = sort_batch(
            test(
                file_dir, buffer_size, data_type=DataType.SEQ),
                batch_size,
                batch_size * 20)
        vocab_size = 0
    return vocab_size, reader


def sort_batch(reader, batch_size, sort_group_size, drop_last=False):
    """
    Create a batched reader.
    :param reader: the data reader to read from.
    :type reader: callable
    :param batch_size: size of each mini-batch
    :type batch_size: int
    :param sort_group_size: size of partial sorted batch
    :type sort_group_size: int
    :param drop_last: drop the last batch, if the size of last batch is not equal to batch_size.
    :type drop_last: bool
    :return: the batched reader.
    :rtype: callable
    """

    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == sort_group_size:
                sortl = sorted(b, key=lambda x: len(x[0]), reverse=True)
                b = []
                c = []
                for sort_i in sortl:
                    c.append(sort_i)
                    if (len(c) == batch_size):
                        yield c
                        c = []
        if drop_last == False and len(b) != 0:
            sortl = sorted(b, key=lambda x: len(x[0]), reverse=True)
            c = []
            for sort_i in sortl:
                c.append(sort_i)
        if (len(c) == batch_size):
            yield c
            c = []

    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size should be a positive integeral value, "
                         "but got batch_size={}".format(batch_size))
    return batch_reader


class DataType(object):
    SEQ = 2

def reader_creator(file_dir, n, data_type):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for l in f:
                    if DataType.SEQ == data_type:
                        l = l.strip().split()
                        l = [w for w in l]
                        src_seq = l[:len(l) - 1]
                        trg_seq = l[1:]
                        if n > 0 and len(src_seq) > n: continue
                        yield src_seq, trg_seq
                    else:
                        assert False, 'error data type'
    return reader

def train(train_dir, n, data_type=DataType.SEQ):
    return reader_creator(train_dir, n, data_type)

def test(test_dir, n, data_type=DataType.SEQ):
    return reader_creator(test_dir, n, data_type)
