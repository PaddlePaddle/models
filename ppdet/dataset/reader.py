# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# function:
#    Interface to build readers for detection data like COCO or VOC
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from . import source
from .transform import transformer as tf
from .transform import operator as op


class Reader(object):
    """ Interface to make readers for training or evaluation
    """

    def __init__(self, data_cf, trans_conf, maxiter=None):
        """ Init
        """
        self._data_cf = data_cf
        self._trans_conf = trans_conf
        self._maxiter = maxiter

    def _make_reader(self, is_train):
        """ Build reader for training or validation
        """
        which = 'train' if is_train else 'val'
        file_conf = self._data_cf[which]

        # 1, Build data source
        samples = -1 if 'samples' not in self._data_cf else self._data_cf[
            'samples']
        sc_conf = {
            'fname': file_conf['anno_file'],
            'image_dir': file_conf['image_dir'],
            'samples': samples
        }
        sc = source.build(sc_conf)

        # 2, Buid a transformed dataset
        ops = self._trans_conf[which]
        gpu_counts = self._trans_conf['gpu_counts']
        batchsize = self._trans_conf['batch_size']
        worker_args = None if 'worker_args' not in \
            self._trans_conf else self._trans_conf['worker_args']
        drop_last = False if 'drop_last' not in \
            self._trans_conf else self._trans_conf['drop_last']
        is_padding = False if 'is_padding' not in \
            self._trans_conf else self._trans_conf['is_padding']
        mapper = op.build(ops)
        mapped_ds = tf.map(sc, mapper, worker_args)
        batched_ds = tf.batch(mapped_ds, batchsize, drop_last, is_padding)

        # 3, Build a reader
        def _reader():
            # TODO: need more elegant code
            maxit = self._maxiter * gpu_counts if self._maxiter else 1
            n = 0
            while n < maxit:
                batched_ds.reset()
                for batch in batched_ds:
                    yield batch
                    n += 1
                    if self._maxiter and n == maxit:
                        return

        return _reader

    def train(self):
        """ Build reader for training
        """
        return self._make_reader(True)

    def val(self):
        """ Build reader for validation
        """
        return self._make_reader(False)
