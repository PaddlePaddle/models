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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import functools
from ...dataset import Dataset


class ProxiedDataset(Dataset):
    """ proxy the method calling to this class to 'self._ds' when itself not avaialbe
    """

    def __init__(self, ds):
        super(ProxiedDataset, self).__init__()
        self._ds = ds
        methods = filter(lambda k: not k.startswith('_'),
                         Dataset.__dict__.keys())

        for m in methods:
            func = functools.partial(self._proxy_method, getattr(self, m))
            setattr(self, m, func)

    def _proxy_method(self, func, *args, **kwargs):
        """ proxy call to 'func', if not available then call self._ds.xxx 
            whose name is the same with func.__name__
        """
        method = func.__name__
        try:
            return func(*args, **kwargs)
        except NotImplementedError as e:
            ds_func = getattr(self._ds, method)
            return ds_func(*args, **kwargs)


class MappedDataset(ProxiedDataset):
    def __init__(self, ds, mapper):
        super(MappedDataset, self).__init__(ds)
        self._ds = ds
        self._mapper = mapper

    def next(self):
        sample = self._ds.next()
        return self._mapper(sample)


class BatchedDataset(ProxiedDataset):
    """ transform samples to batches
    """

    def __init__(self,
                 ds,
                 batchsize,
                 coarsest_stride,
                 drop_last=True,
                 is_padding=False):
        """
        Args:
            ds (instance of Dataset): dataset to be batched
            batchsize (int): sample number for each batch
            coarsest_stride (int): stride of the coarsest FPN level
            drop_last (bool): whether to drop last samples when not
                enough for one batch
        """
        super(BatchedDataset, self).__init__(ds)
        self._batchsz = batchsize
        self._stride = coarsest_stride
        self._drop_last = drop_last
        self.is_padding = is_padding

    def padding_minibatch(self, batch_data):
        if len(batch_data) == 1 and self._stride == 1:
            return batch_data
        max_shape = np.array([data[0].shape for data in batch_data]).max(axis=0)
        if self._stride > 1:
            max_shape[1] = int(
                np.ceil(max_shape[1] / self._stride) * self._stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / self._stride) * self._stride)
        padding_batch = []
        for data in batch_data:
            im_c, im_h, im_w = data[0].shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = data[0]
            data[1][:2] = max_shape[1:3]
            padding_batch.append((padding_im, ) + data[1:])
        return padding_batch

    def next(self):
        """ proxy to self._ds.next
        """

        def has_empty(items):
            if any(x is None for x in items):
                return True
            if any(x.size == 0 for x in items):
                return True
            return False

        batch = []
        for _ in range(self._batchsz):
            try:
                out = self._ds.next()
                while has_empty(out):
                    out = self._ds.next()
                batch.append(out)
            except StopIteration as e:
                if not self._drop_last and len(batch) > 0:
                    if self.is_padding:
                        batch = self.padding_minibatch(batch)
                    return batch
                else:
                    raise StopIteration
        if self.is_padding:
            batch = self.padding_minibatch(batch)
        return batch
