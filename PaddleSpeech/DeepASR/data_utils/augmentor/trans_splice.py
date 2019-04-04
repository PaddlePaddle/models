from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math


class TransSplice(object):
    """ copy feature context to construct new feature
        expand feature data from shape (frame_num, frame_dim) 
        to shape (frame_num, frame_dim * 11)

        Attributes:
            _nleft_context(int): copy left context number
            _nright_context(int): copy right context number
    """

    def __init__(self, nleft_context=5, nright_context=5):
        """ init construction
            Args:
                nleft_context(int):
                nright_context(int):
        """
        self._nleft_context = nleft_context
        self._nright_context = nright_context

    def perform_trans(self, sample):
        """ copy feature context 
        Args:
            sample(object): input sample(feature, label)
        Return:
            (feature, label, name)
        """
        (feature, label, name) = sample
        nframe_num = feature.shape[0]
        nframe_dim = feature.shape[1]
        nnew_frame_dim = nframe_dim * (
            self._nleft_context + self._nright_context + 1)
        mat = np.zeros(
            (nframe_num + self._nleft_context + self._nright_context,
             nframe_dim),
            dtype="float32")
        ret = np.zeros((nframe_num, nnew_frame_dim), dtype="float32")

        #copy left
        for i in xrange(self._nleft_context):
            mat[i, :] = feature[0, :]

        #copy middle 
        mat[self._nleft_context:self._nleft_context +
            nframe_num, :] = feature[:, :]

        #copy right
        for i in xrange(self._nright_context):
            mat[i + self._nleft_context + nframe_num, :] = feature[-1, :]

        mat = mat.reshape(mat.shape[0] * mat.shape[1])
        ret = ret.reshape(ret.shape[0] * ret.shape[1])
        for i in xrange(nframe_num):
            np.copyto(ret[i * nnew_frame_dim:(i + 1) * nnew_frame_dim],
                      mat[i * nframe_dim:i * nframe_dim + nnew_frame_dim])
        ret = ret.reshape((nframe_num, nnew_frame_dim))
        return (ret, label, name)
