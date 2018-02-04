from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def lodtensor_to_ndarray(lod_tensor):
    dims = lod_tensor.get_dims()
    ret = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ret.ravel()[i] = lod_tensor.get_float_element(i)
    return ret, lod_tensor.lod()
