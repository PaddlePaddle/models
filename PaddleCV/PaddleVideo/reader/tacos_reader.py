#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import cv2
import math
import random
import functools
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import paddle
from PIL import Image, ImageEnhance
import logging

from .reader_utils import DataReader

class TacosReader(DataReader):

    def __init__(self, name, mode, cfg):
	self.name = name
	self.mode = mode
	self.cfg = cfg
    def create_reader(self):
        cfg = self.cfg
        mode = self.mode
        num_reader_threads = cfg[mode.upper()]['num_reader_threads']
        assert num_reader_threads >=1, \
                "number of reader threads({}) should be a positive integer".format(num_reader_threads)
        if num_reader_threads == 1:
            reader_func = make_reader
        else:
            reader_func = make_multi_reader
	
	filelist = cfg[mode.upper()]['']
	if self.mode == 'train':
	    return reader_func()
	elif self.mode == 'valid':
	    return reader_func()
	else:
	    logger.info("Not implemented")
	    raise NotImplementedError

def make_reader(cfg):
    def reader():
	cs = cPickle.load(open(cfg.TRAIN.train_clip_sentvec))
        movie_length_info = cPickle.load(open(cfg.TRAIN.movie_length_info))

   #put train() in here


