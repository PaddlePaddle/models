from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

def accuracy(targets, preds):
    """Get the class-level top1 and top5 of model.

    Usage:

    .. code-blcok::python

        top1, top5 = accuracy(targets, preds)

    :params args: evaluate the prediction of model.
    :type args: numpy.array

    """
    top1 = np.zeros((5000,), dtype=np.float32)
    top5 = np.zeros((5000,), dtype=np.float32)
    count = np.zeros((5000,), dtype=np.float32)

    for index in range(targets.shape[0]):
        target = targets[index]
        if target == preds[index,0]:
            top1[target] += 1
            top5[target] += 1
        elif np.sum(target == preds[index,:5]):
            top5[target] += 1

        count[target] += 1
    return (top1/(count+1e-12)).mean(), (top5/(count+1e-12)).mean()
