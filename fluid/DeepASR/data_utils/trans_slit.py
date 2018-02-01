#by zhxfl
import numpy
import math


class TransSplit(object):
    """ expand feature data from shape (frame_num, frame_dim) 
        to shape (frame_num, frame_dim * 11)
        
    """

    def __init__(self, nleft_context=5, nright_context=5):
        self._nleft_context = nleft_context
        self._nright_context = nright_context
