import argparse
import gzip
import logging
import sys
import numpy

__all__ = [
    "open_file",
    "cumsum",
    "logger",
    "DotBar",
    "load_dict",
    "load_wordvecs",
]

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def open_file(filename, *args1, **args2):
    """
    Open a file

    :param filename: name of the file
    :type filename: str
    :return: a file handler
    """
    if filename.endswith(".gz"):
        return gzip.open(filename, *args1, **args2)
    else:
        return open(filename, *args1, **args2)


def cumsum(array):
    """
    Caculute the accumulated sum of array. For example, array=[1, 2, 3], the
    result is [1, 1+2, 1+2+3]

    :param array: input array
    :type array: python list or numpy array
    :return: the accumulated sum of array
    """
    if len(array) <= 1:
        return list(array)
    ret = list(array)
    for i in xrange(1, len(ret)):
        ret[i] += ret[i - 1]
    return ret


class DotBar(object):
    """
    A simple dot bar
    """

    def __init__(self, obj, step=200, dots_per_line=50, f=sys.stderr):
        """
        :param obj: an iteratable obj
        :type obj: a python itertor
        :param step: print a dot every step iterations
        :type step: int
        :param dots_per_line: dots each line
        :type dots_per_line: int
        :param f: print dot to f, default value is sys.stderr
        :type f: a file handler
        """
        self.obj = obj
        self.step = step
        self.dots_per_line = dots_per_line
        self.f = f

    def __enter__(self, ):
        self.obj.__enter__()
        self.idx = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.write("\n")
        if self.obj is sys.stdin or self.obj is sys.stdout:
            return
        self.obj.__exit__(exc_type, exc_value, traceback)

    def __iter__(self):
        return self

    def next(self):
        self.idx += 1
        if self.idx % self.step == 0:
            self.f.write(".")
        if self.idx % (self.step * self.dots_per_line) == 0:
            self.f.write("\n")

        return self.obj.next()


def load_dict(word_dict_path):
    with open_file(word_dict_path) as f:
        # the first word must be OOV
        vocab = {k.rstrip("\n").split()[0].decode("utf-8"):i \
                        for i, k in enumerate(f)}
    return vocab


def load_wordvecs(word_dict_path, wordvecs_path):
    vocab = load_dict(word_dict_path)
    wordvecs = numpy.loadtxt(wordvecs_path, delimiter=",", dtype="float32")
    assert len(vocab) == wordvecs.shape[0]
    return vocab, wordvecs
