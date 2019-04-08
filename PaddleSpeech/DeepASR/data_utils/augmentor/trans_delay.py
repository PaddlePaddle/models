from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math


class TransDelay(object):
    """ Delay label, and copy first label value in the front. 
        Attributes:
            _delay_time : the delay frame num of label 
    """

    def __init__(self, delay_time):
        """init construction
            Args:
                delay_time : the delay frame num of label
        """
        self._delay_time = delay_time

    def perform_trans(self, sample):
        """ 
            Args:
                sample(object):input sample, contain feature numpy and label numpy, sample name list
            Returns:
                (feature, label, name)
        """
        (feature, label, name) = sample

        shape = label.shape
        assert len(shape) == 2
        label[self._delay_time:shape[0]] = label[0:shape[0] - self._delay_time]
        for i in xrange(self._delay_time):
            label[i][0] = label[self._delay_time][0]

        return (feature, label, name)
