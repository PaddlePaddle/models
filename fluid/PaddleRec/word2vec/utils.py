import sys
import collections
import six
import time
import numpy as np
import paddle.fluid as fluid
import paddle
import os
import preprocess


def BuildWord_IdMap(dict_path):
    word_to_id = dict()
    id_to_word = dict()
    with open(dict_path, 'r') as f:
        for line in f:
            word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
            id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]
    return word_to_id, id_to_word


def prepare_data(file_dir, dict_path, batch_size):
    w2i, i2w = BuildWord_IdMap(dict_path)
    vocab_size = len(i2w)
    reader = paddle.batch(test(file_dir, w2i), batch_size)
    return vocab_size, reader, i2w


def native_to_unicode(s):
    if _is_unicode(s):
        return s
    try:
        return _to_unicode(s)
    except UnicodeDecodeError:
        res = _to_unicode(s, ignore_errors=True)
        return res


def _is_unicode(s):
    if six.PY2:
        if isinstance(s, unicode):
            return True
    else:
        if isinstance(s, str):
            return True
    return False


def _to_unicode(s, ignore_errors=False):
    if _is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def strip_lines(line, vocab):
    return _replace_oov(vocab, native_to_unicode(line))


def _replace_oov(original_vocab, line):
    """Replace out-of-vocab words with "<UNK>".
  This maintains compatibility with published results.
  Args:
    original_vocab: a set of strings (The standard vocabulary for the dataset)
    line: a unicode string - a space-delimited sequence of words.
  Returns:
    a unicode string - a space-delimited sequence of words.
  """
    return u" ".join([
        word if word in original_vocab else u"<UNK>" for word in line.split()
    ])


def reader_creator(file_dir, word_to_id):
    def reader():
        files = os.listdir(file_dir)
        for fi in files:
            with open(file_dir + '/' + fi, "r") as f:
                for line in f:
                    if ':' in line:
                        pass
                    else:
                        line = strip_lines(line.lower(), word_to_id)
                        line = line.split()
                        yield [word_to_id[line[0]]], [word_to_id[line[1]]], [
                            word_to_id[line[2]]
                        ], [word_to_id[line[3]]], [
                            word_to_id[line[0]], word_to_id[line[1]],
                            word_to_id[line[2]]
                        ]

    return reader


def test(test_dir, w2i):
    return reader_creator(test_dir, w2i)
