#by zhxfl 2018.01.31
import numpy
import math

class TransSplice(object):
    """ expand feature data from shape (frame_num, frame_dim) 
        to shape (frame_num, frame_dim * 11)
        
    """
    def __init__(self, nleft_context = 5, nright_context = 5):
        """ init construction
        """
        self._nleft_context = nleft_context
        self._nright_context = nright_context

    def perform_trans(self, sample):
        """ splice
        """
        (feature, label) = sample
        nframe_num = feature.shape[0]
        nframe_dim = feature.shape[1]
        nnew_frame_dim = nframe_dim * (self._nleft_context + self._nright_context + 1)
        mat = numpy.zeros((nframe_num + self._nleft_context + self._nright_context, nframe_dim), dtype="float32")
        ret = numpy.zeros((nframe_num, nnew_frame_dim), dtype="float32")

        #copy left
        for i in xrange(self._nleft_context):
            mat[i,:] = feature[0,:]

        #copy middle 
        mat[self._nleft_context: self._nleft_context + nframe_num,:] = feature[:,:]

        #copy right
        for i in xrange(self._nright_context):
            mat[i + self._nleft_context + nframe_num,:] = feature[-1,:]

        mat = mat.reshape(mat.shape[0] * mat.shape[1])
        ret = ret.reshape(ret.shape[0] * ret.shape[1])
        for i in xrange(nframe_num):
            numpy.copyto(ret[i * nnew_frame_dim : (i + 1) * nnew_frame_dim],
                    mat[i * nframe_dim: i * nframe_dim + nnew_frame_dim])
        ret = ret.reshape((nframe_num, nnew_frame_dim))
        return (ret, label)

