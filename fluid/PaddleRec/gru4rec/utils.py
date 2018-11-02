import sys
import collections
import six
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


def prepare_data(train_filename,
                 test_filename,
                 batch_size,
                 buffer_size=1000,
                 word_freq_threshold=0,
                 enable_ce=False):
    """ prepare the English Pann Treebank (PTB) data """
    print("start constuct word dict")
    vocab = build_dict(word_freq_threshold, train_filename, test_filename)
    print("construct word dict done\n")
    if enable_ce:
        train_reader = paddle.batch(
            train(
                train_filename, vocab, buffer_size, data_type=DataType.SEQ),
            batch_size)
    else:
        train_reader = sort_batch(
            paddle.reader.shuffle(
                train(
                    train_filename, vocab, buffer_size, data_type=DataType.SEQ),
                buf_size=buffer_size),
            batch_size,
            batch_size * 20)
    test_reader = sort_batch(
        test(
            test_filename, vocab, buffer_size, data_type=DataType.SEQ),
        batch_size,
        batch_size * 20)
    return vocab, train_reader, test_reader


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


def word_count(input_file, word_freq=None):
    """
    compute word count from corpus
    """
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for l in input_file:
        for w in l.strip().split():
            word_freq[w] += 1

    return word_freq


def build_dict(min_word_freq=50, train_filename="", test_filename=""):
    """
    Build a word dictionary from the corpus,  Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    with open(train_filename) as trainf:
        with open(test_filename) as testf:
            word_freq = word_count(testf, word_count(trainf))

    word_freq = [x for x in six.iteritems(word_freq) if x[1] > min_word_freq]
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(list(zip(words, six.moves.range(len(words)))))
    return word_idx


def reader_creator(filename, word_idx, n, data_type):
    def reader():
        with open(filename) as f:
            for l in f:
                if DataType.SEQ == data_type:
                    l = l.strip().split()
                    l = [word_idx.get(w) for w in l]
                    src_seq = l[:len(l) - 1]
                    trg_seq = l[1:]
                    if n > 0 and len(src_seq) > n: continue
                    yield src_seq, trg_seq
                else:
                    assert False, 'error data type'

    return reader


def train(filename, word_idx, n, data_type=DataType.SEQ):
    return reader_creator(filename, word_idx, n, data_type)


def test(filename, word_idx, n, data_type=DataType.SEQ):
    return reader_creator(filename, word_idx, n, data_type)
