import re
import sys
import collections
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

def prepare_data(train_filename,
                 test_filename,
                 batch_size,
                 neg_size=1,
                 buffer_size=1000,
                 word_freq_threshold=0,
                 enable_ce=False):
    """ prepare the AG's News Topic Classification data """
    print("start constuct word dict")
    vocab_text = build_dict(2, word_freq_threshold, train_filename, test_filename)
    vocab_tag = build_dict(0, word_freq_threshold, train_filename, test_filename)
    print("construct word dict done\n")
    train_reader = sort_batch(
        paddle.reader.shuffle(
            train(
                train_filename, vocab_text, vocab_tag, buffer_size, data_type=DataType.SEQ),
            buf_size=buffer_size),
        batch_size, batch_size * 20)
    test_reader = sort_batch(
        test(
            test_filename, vocab_text, vocab_tag, buffer_size, data_type=DataType.SEQ),
        batch_size, batch_size * 20)
    return vocab_text, vocab_tag, train_reader, test_reader

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

def word_count(column_num, input_file, word_freq=None):
    """
    compute word count from corpus
    """
    if word_freq is None:
        word_freq = collections.defaultdict(int)
    data_file = csv.reader(input_file)
    for row in data_file:
        for w in re.split(r'\W+',row[column_num].strip()):
            word_freq[w]+= 1
    return word_freq

def build_dict(column_num=2, min_word_freq=50, train_filename="", test_filename=""):
    """
    Build a word dictionary from the corpus,  Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    with open(train_filename) as trainf:
        with open(test_filename) as testf:
            word_freq = word_count(column_num, testf, word_count(column_num, trainf))

    word_freq = [x for x in six.iteritems(word_freq) if x[1] > min_word_freq]
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(list(zip(words, six.moves.range(len(words)))))
    return word_idx

def reader_creator(filename, text_idx, tag_idx, n, data_type):
    def reader():
        with open(filename) as input_file:
            data_file = csv.reader(input_file)
            for row in data_file:
                text_raw = re.split(r'\W+', row[2].strip())
                text = [text_idx.get(w) for w in text_raw]
                tag_raw = re.split(r'\W+', row[0].strip())
                pos_index = tag_idx.get(tag_raw[0])
                pos_tag=[]
                pos_tag.append(pos_index)
                neg_tag=[]
                max_iter = 100
                now_iter = 0
                sum_n = 0
                while(sum_n < 1) :
                    now_iter += 1
                    if now_iter > max_iter:
                        print("error : only one class")
                        sys.exit(0)
                    rand_i = np.random.randint(0, len(tag_idx))
                    if rand_i != pos_index:
                        neg_index=rand_i
                        neg_tag.append(neg_index)
                        sum_n += 1
                if n > 0 and len(text) > n: continue
                yield text, pos_tag, neg_tag
    return reader

def train(filename, text_idx, tag_idx, n, data_type=DataType.SEQ):
    return reader_creator(filename, text_idx, tag_idx, n, data_type)

def test(filename, text_idx, tag_idx, n, data_type=DataType.SEQ):
    return reader_creator(filename, text_idx, tag_idx, n, data_type)
