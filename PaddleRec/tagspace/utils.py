import re
import sys
import collections
import os
import six
import time
import numpy as np
import paddle.fluid as fluid
import paddle
import csv

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
                 vocab_text_path,
                 vocab_tag_path,
                 batch_size,
                 neg_size,
                 buffer_size,
                 is_train=True):
    """ prepare the AG's News Topic Classification data """
    print("start read file")
    if is_train:
        vocab_text_size = get_vocab_size(vocab_text_path)
        vocab_tag_size = get_vocab_size(vocab_tag_path)
        reader = sort_batch(
            paddle.reader.shuffle(
                train(
                    file_dir, vocab_tag_size, neg_size,
                    buffer_size, data_type=DataType.SEQ),
                buf_size=buffer_size),
            batch_size, batch_size * 20)
    else:
        vocab_tag_size = get_vocab_size(vocab_tag_path)
        vocab_text_size = 0
        reader = sort_batch(
            test(
                file_dir, vocab_tag_size, buffer_size, data_type=DataType.SEQ),
            batch_size, batch_size * 20)
    return vocab_text_size, vocab_tag_size, reader

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

def train_reader_creator(file_dir, tag_size, neg_size, n, data_type):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for l in f:
                    l = l.strip().split(",")
                    pos_index = int(l[0])
                    pos_tag = []
                    pos_tag.append(pos_index)
                    text_raw = l[1].split()
                    text = [int(w) for w in text_raw]
                    neg_tag = []
                    max_iter = 100
                    now_iter = 0
                    sum_n = 0
                    while(sum_n < neg_size) :
                        now_iter += 1
                        if now_iter > max_iter:
                            print("error : only one class")
                            sys.exit(0)
                        rand_i = np.random.randint(0, tag_size)
                        if rand_i != pos_index:
                            neg_index = rand_i
                            neg_tag.append(neg_index)
                            sum_n += 1
                    if n > 0 and len(text) > n: continue
                    yield text, pos_tag, neg_tag
    return reader

def test_reader_creator(file_dir, tag_size, n, data_type):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for l in f:
                    l = l.strip().split(",")
                    pos_index = int(l[0])
                    pos_tag = []
                    pos_tag.append(pos_index)
                    text_raw = l[1].split()
                    text = [int(w) for w in text_raw]
                    for ii in range(tag_size):
                        tag = []
                        tag.append(ii)
                        yield text, tag, pos_tag
    return reader


def train(train_dir, tag_size, neg_size, n, data_type=DataType.SEQ):
    return train_reader_creator(train_dir, tag_size, neg_size, n, data_type)

def test(test_dir, tag_size, n, data_type=DataType.SEQ):
    return test_reader_creator(test_dir, tag_size, n, data_type)
