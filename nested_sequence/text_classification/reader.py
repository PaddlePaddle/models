"""
IMDB dataset.

This module downloads IMDB dataset from
http://ai.stanford.edu/%7Eamaas/data/sentiment/. This dataset contains a set
of 25,000 highly polar movie reviews for training, and 25,000 for testing.
Besides, this module also provides API for building dictionary.
"""
import collections
import tarfile
import Queue
import re
import string
import threading
import os

import paddle.v2.dataset.common

URL = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
MD5 = '7c2ac02c03563afcf9b574c7e56c153a'


def tokenize(pattern):
    """
    Read files that match the given pattern.  Tokenize and yield each file.
    """
    with tarfile.open(paddle.v2.dataset.common.download(URL, 'imdb',
                                                        MD5)) as tarf:
        tf = tarf.next()
        while tf != None:
            if bool(pattern.match(tf.name)):
                # newline and punctuations removal and ad-hoc tokenization.
                docs = tarf.extractfile(tf).read().rstrip("\n\r").lower().split(
                    '.')
                doc_list = []
                for doc in docs:
                    doc = doc.strip()
                    if doc:
                        doc_without_punc = doc.translate(
                            None, string.punctuation).strip()
                        if doc_without_punc:
                            doc_list.append(
                                [word for word in doc_without_punc.split()])
                yield doc_list
            tf = tarf.next()


def imdb_build_dict(pattern, cutoff):
    """
    Build a word dictionary from the corpus. Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    word_freq = collections.defaultdict(int)
    for doc_list in tokenize(pattern):
        for doc in doc_list:
            for word in doc:
                word_freq[word] += 1

    word_freq['<unk>'] = cutoff + 1
    word_freq = filter(lambda x: x[1] > cutoff, word_freq.items())
    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    word_idx = dict(zip(words, xrange(len(words))))
    return word_idx


def reader_creator(pos_pattern, neg_pattern, word_idx, buffer_size):
    UNK = word_idx['<unk>']

    qs = [Queue.Queue(maxsize=buffer_size), Queue.Queue(maxsize=buffer_size)]

    def load(pattern, queue):
        for doc_list in tokenize(pattern):
            queue.put(doc_list)
        queue.put(None)

    def reader():
        # Creates two threads that loads positive and negative samples
        # into qs.
        t0 = threading.Thread(
            target=load, args=(
                pos_pattern,
                qs[0], ))
        t0.daemon = True
        t0.start()

        t1 = threading.Thread(
            target=load, args=(
                neg_pattern,
                qs[1], ))
        t1.daemon = True
        t1.start()

        # Read alternatively from qs[0] and qs[1].
        i = 0
        doc_list = qs[i].get()

        while doc_list != None:
            ids_list = []
            for doc in doc_list:
                ids_list.append([word_idx.get(w, UNK) for w in doc])
            yield ids_list, i % 2
            i += 1
            doc_list = qs[i % 2].get()

        # If any queue is empty, reads from the other queue.
        i += 1
        doc_list = qs[i % 2].get()
        while doc_list != None:
            ids_list = []
            for doc in doc_list:
                ids_list.append([word_idx.get(w, UNK) for w in doc])
            yield ids_list, i % 2
            doc_list = qs[i % 2].get()

    return reader()


def imdb_train(word_idx):
    """
    IMDB training set creator.

    It returns a reader creator, each sample in the reader is an zero-based ID
    subsequence and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        re.compile("aclImdb/train/pos/.*\.txt$"),
        re.compile("aclImdb/train/neg/.*\.txt$"), word_idx, 1000)


def imdb_test(word_idx):
    """
    IMDB test set creator.

    It returns a reader creator, each sample in the reader is an zero-based ID
    subsequence and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Test reader creator
    :rtype: callable
    """
    return reader_creator(
        re.compile("aclImdb/test/pos/.*\.txt$"),
        re.compile("aclImdb/test/neg/.*\.txt$"), word_idx, 1000)


def imdb_word_dict():
    """
    Build a word dictionary from the corpus.

    :return: Word dictionary
    :rtype: dict
    """
    return imdb_build_dict(
        re.compile("aclImdb/((train)|(test))/((pos)|(neg))/.*\.txt$"), 150)


def train_reader(data_dir, word_dict, label_dict):
    """
    Reader interface for training data

    :param data_dir: data directory
    :type data_dir: str
    :param word_dict: path of word dictionary,
        the dictionary must has a "UNK" in it.
    :type word_dict: Python dict
    :param label_dict: path of label dictionary.
    :type label_dict: Python dict
    """

    def reader():
        UNK_ID = word_dict['<unk>']
        word_col = 1
        lbl_col = 0

        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, "r") as f:
                for line in f:
                    line_split = line.strip().split("\t")
                    doc = line_split[word_col]
                    doc_ids = []
                    for sent in doc.strip().split("."):
                        sent_ids = [
                            word_dict.get(w, UNK_ID) for w in sent.split()
                        ]
                        if sent_ids:
                            doc_ids.append(sent_ids)

                    yield doc_ids, label_dict[line_split[lbl_col]]

    return reader


def infer_reader(file_path, word_dict):
    """
    Reader interface for prediction

    :param data_dir: data directory
    :type data_dir: str
    :param word_dict: path of word dictionary,
        the dictionary must has a "UNK" in it.
    :type word_dict: Python dict
    """

    def reader():
        UNK_ID = word_dict['<unk>']

        with open(file_path, "r") as f:
            for doc in f:
                doc_ids = []
                for sent in doc.strip().split("."):
                    sent_ids = [word_dict.get(w, UNK_ID) for w in sent.split()]
                    if sent_ids:
                        doc_ids.append(sent_ids)

                yield doc_ids, doc

    return reader
