from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from six import reraise
from tblib import Traceback

import numpy as np


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


def lodtensor_to_ndarray(lod_tensor):
    """conver lodtensor to ndarray
    """
    dims = lod_tensor.get_dims()
    ret = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ret.ravel()[i] = lod_tensor.get_float_element(i)
    return ret, lod_tensor.lod()


def suppress_signal(signo, stack_frame):
    pass


def suppress_complaints(verbose):
    def decorator_maker(func):
        def suppress_warpper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except:
                et, ev, tb = sys.exc_info()
                tb = Traceback(tb)
                if verbose == 1:
                    reraise(et, ev, tb.as_traceback())

        return suppress_warpper

    return decorator_maker
