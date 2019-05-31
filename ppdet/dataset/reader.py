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

import logging

from . import source
from .transform import transformer as tf
from .transform import operator as op

logger = logging.getLogger(__name__)

class Reader(object):
    """ Interface to make readers for training or evaluation
    """

    def __init__(self, data_cf, trans_conf, maxiter=None):
        """ Init
        """
        self._data_cf = data_cf
        self._trans_conf = trans_conf
        self._maxiter = maxiter

    def _make_reader(self, which):
        """ Build reader for training or validation
        """
        file_conf = self._data_cf[which]

        # 1, Build data source
        samples = -1 if 'SAMPLES' not in file_conf else file_conf['SAMPLES']
        is_shuffle = True if 'IS_SHUFFLE' not in file_conf \
            else file_conf['IS_SHUFFLE']
        sc_conf = {
            'fname': file_conf['ANNO_FILE'],
            'image_dir': file_conf['IMAGE_DIR'],
            'samples': samples,
            'is_shuffle': is_shuffle
        }
        sc = source.build(sc_conf)

        # 2, Buid a transformed dataset
        ops = self._trans_conf[which]['OPS']
        batchsize = self._trans_conf[which]['BATCH_SIZE']
        worker_args = None if 'WORKER_CONF' not in \
            self._trans_conf[which] else self._trans_conf[which]['WORKER_CONF']

        drop_last = False if 'DROP_LAST' not in \
            self._trans_conf[which] else self._trans_conf[which]['DROP_LAST']
        is_padding = False if 'IS_PADDING' not in \
            self._trans_conf[which] else self._trans_conf[which]['IS_PADDING']
        coarsest_stride = 1 if 'COAREST_STRIDE' not in \
            self._trans_conf[which] else self._trans_conf[which]['COAREST_STRIDE']
        mapper = op.build(ops)

        worker_args = {k.lower(): v for k, v in worker_args.items()}
        mapped_ds = tf.map(sc, mapper, worker_args)
        batched_ds = tf.batch(mapped_ds, batchsize, coarsest_stride, drop_last,
                              is_padding)

        # 3, Build a reader
        def _reader():
            maxit = self._maxiter if self._maxiter else 1
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
        return self._make_reader('TRAIN')

    def val(self):
        """ Build reader for validation
        """
        return self._make_reader('VAL')

    def test(self):
        """ Build reader for inference
        """
        return self._make_reader('TEST')
