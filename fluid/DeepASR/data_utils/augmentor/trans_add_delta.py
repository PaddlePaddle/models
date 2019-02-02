from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import copy


class TransAddDelta(object):
    """ add delta of feature data 
        trans feature for shape(a, b) to shape(a, b * 3)

        Attributes:
            _norder(int):
            _window(int):
    """

    def __init__(self, norder=2, nwindow=2):
        """ init construction
            Args:
                norder: default 2 
                nwindow: default 2
        """
        self._norder = norder
        self._nwindow = nwindow

    def perform_trans(self, sample):
        """ add delta for feature
            trans feature shape from (a,b) to (a, b * 3)

            Args: 
                sample(object,tuple): contain feature numpy and label numpy
            Returns:
                (feature, label, name)
        """
        (feature, label, name) = sample
        frame_dim = feature.shape[1]
        d_frame_dim = frame_dim * 3
        head_filled = 5
        tail_filled = 5
        mat = np.zeros(
            (feature.shape[0] + head_filled + tail_filled, d_frame_dim),
            dtype="float32")
        #copy first frame
        for i in xrange(head_filled):
            np.copyto(mat[i, 0:frame_dim], feature[0, :])

        np.copyto(mat[head_filled:head_filled + feature.shape[0], 0:frame_dim],
                  feature[:, :])

        # copy last frame
        for i in xrange(head_filled + feature.shape[0], mat.shape[0], 1):
            np.copyto(mat[i, 0:frame_dim], feature[feature.shape[0] - 1, :])

        nframe = feature.shape[0]
        start = head_filled
        tmp_shape = mat.shape
        mat = mat.reshape((tmp_shape[0] * tmp_shape[1]))
        self._regress(mat, start * d_frame_dim, mat,
                      start * d_frame_dim + frame_dim, frame_dim, nframe,
                      d_frame_dim)
        self._regress(mat, start * d_frame_dim + frame_dim, mat,
                      start * d_frame_dim + 2 * frame_dim, frame_dim, nframe,
                      d_frame_dim)
        mat.shape = tmp_shape
        return (mat[head_filled:mat.shape[0] - tail_filled, :], label, name)

    def _regress(self, data_in, start_in, data_out, start_out, size, n, step):
        """ regress
            Args:
                data_in: in data
                start_in: start index of data_in
                data_out: out data
                start_out: start index of data_out
                size: frame dimentional
                n: frame num
                step: 3 * (frame num)
            Returns:
                None
        """
        sigma_t2 = 0.0
        delta_window = self._nwindow
        for t in xrange(1, delta_window + 1):
            sigma_t2 += t * t

        sigma_t2 *= 2.0
        for i in xrange(n):
            fp1 = start_in
            fp2 = start_out
            for j in xrange(size):
                back = fp1
                forw = fp1
                sum = 0.0
                for t in xrange(1, delta_window + 1):
                    back -= step
                    forw += step
                    sum += t * (data_in[forw] - data_in[back])

                data_out[fp2] = sum / sigma_t2
                fp1 += 1
                fp2 += 1
            start_in += step
            start_out += step
