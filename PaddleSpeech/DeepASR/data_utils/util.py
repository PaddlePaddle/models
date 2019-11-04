from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from six import reraise
from tblib import Traceback

import numpy as np


def lodtensor_to_ndarray(result):
    return np.array(result), result.lod()


def to_lodtensor(data, place):
    """convert tensor to lodtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = numpy.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def split_infer_result(infer_seq, lod):
    infer_batch = []
    for i in xrange(0, len(lod[0]) - 1):
        infer_batch.append(infer_seq[lod[0][i]:lod[0][i + 1]])
    return infer_batch


class CriticalException(Exception):
    pass


def suppress_signal(signo, stack_frame):
    pass


def suppress_complaints(verbose, notify=None):
    def decorator_maker(func):
        def suppress_warpper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except:
                et, ev, tb = sys.exc_info()

                if notify is not None:
                    notify(except_type=et, except_value=ev, traceback=tb)

                if verbose == 1 or isinstance(ev, CriticalException):
                    reraise(et, ev, Traceback(tb).as_traceback())

        return suppress_warpper

    return decorator_maker


class ForceExitWrapper(object):
    def __init__(self, exit_flag):
        self._exit_flag = exit_flag

    @suppress_complaints(verbose=0)
    def __call__(self, *args, **kwargs):
        self._exit_flag.value = True

    def __eq__(self, flag):
        return self._exit_flag.value == flag
