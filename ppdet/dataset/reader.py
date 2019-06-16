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
from . import transform as tf

logger = logging.getLogger(__name__)


class Reader(object):
    """ Interface to make readers for training or evaluation
    """

    def __init__(self, data_cf, trans_conf, maxiter=-1):
        """ Init
        """
        self._data_cf = data_cf
        self._trans_conf = trans_conf
        self._maxiter = maxiter
        assert type(self._maxiter
                    ) is int or long, 'The type of maxiter is not int or long.'
        self._cname2cid = None

    def _make_reader(self, which):
        """ Build reader for training or validation
        """
        file_conf = self._data_cf[which]

        # 1, Build data source

        sc_conf = {'data_cf': file_conf, 'cname2cid': self._cname2cid}
        sc = source.build(sc_conf)

        # 2, Buid a transformed dataset
        ops = self._trans_conf[which]['OPS']
        batchsize = self._trans_conf[which]['BATCH_SIZE']
        drop_last = False if 'DROP_LAST' not in \
            self._trans_conf[which] else self._trans_conf[which]['DROP_LAST']

        mapper = tf.build(ops, {'is_train': which == 'TRAIN'})

        worker_args = None
        if 'WORKER_CONF' in self._trans_conf:
            worker_args = self._trans_conf['WORKER_CONF']
            worker_args = {k.lower(): v for k, v in worker_args.items()}

        mapped_ds = tf.map(sc, mapper, worker_args)
        batched_ds = tf.batch(mapped_ds, batchsize, drop_last)

        trans_conf = {k.lower(): v for k, v in self._trans_conf[which].items()}
        need_keys = {
            'is_padding', 'coarsest_stride', 'random_shapes', 'multi_scales'
        }
        bm_config = {
            key: value
            for key, value in trans_conf.items() if key in need_keys
        }
        batched_ds = tf.batch_map(batched_ds, bm_config)

        batched_ds.reset()
        if which.lower() == 'train':
            if self._cname2cid is not None:
                logger.warn(
                    'The cname2cid field has been setted, and it will be overrided by a new one.'
                )
            self._cname2cid = sc.cname2cid

        # 3, Build a reader
        maxit = -1 if self._maxiter <= 0 else self._maxiter

        def _reader():
            n = 0
            while True:
                for batch in batched_ds:
                    yield batch
                    n += 1
                    if maxit > 0 and n == maxit:
                        return
                batched_ds.reset()
                if maxit <= 0:
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
