"""
imikolov's simple dataset.
This module will download dataset from
http://www.fit.vutbr.cz/~imikolov/rnnlm/ and parse training set and test set
into paddle reader creators.
"""

from __future__ import print_function

import paddle.dataset.common
import collections
import tarfile
import six

__all__ = ['train', 'test', 'build_dict', 'convert']



class DataType(object):
    SEQ = 2


def word_count(f, word_freq=None):
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for l in f:
        for w in l.strip().split():
            word_freq[w] += 1

    return word_freq


def build_dict(min_word_freq=50,train_filename="",test_filename=""):
    """
    Build a word dictionary from the corpus,  Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    with open(train_filename) as trainf:
	with open(test_filename) as testf:
	    word_freq = word_count(testf, word_count(trainf))
            
    if '<unk>' in word_freq:
            # remove <unk> for now, since we will set it as last index
        del word_freq['<unk>']

    word_freq = [
            x for x in six.iteritems(word_freq) if x[1] > min_word_freq
        ]
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
			src_seq = l[:len(l)-1]
			trg_seq = l[1:]
                    	if n > 0 and len(src_seq) > n: continue
                    	yield src_seq, trg_seq
                else:
                    	assert False, 'error data type'

    return reader


def train(filename,word_idx, n, data_type=DataType.SEQ):
    return reader_creator(filename, word_idx, n,
                          data_type)
def test(filename,word_idx, n, data_type=DataType.SEQ):
    return reader_creator(filename, word_idx, n,
                          data_type)
