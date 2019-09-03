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

import numbers
import os
import random

try:
    from collections.abc import Mapping
except Exception:
    from collections import Mapping

import numpy as np

from paddle import fluid
from ppdet.core.workspace import register, serializable

from . import datasets
from . import samplers
from . import transforms
from . import dataloader

for m in [datasets, samplers, transforms]:
    for c in getattr(m, '__all__'):
        serializable(register(getattr(m, c)))

type_map = {
    'coco': datasets.COCODataSet,
    'voc': datasets.PascalVocDataSet,
    'folder': datasets.ImageFolder,
}


class ToFeedDict(object):
    def __init__(self,
                 feed_vars=[],
                 extra_vars=[],
                 pin_memory=False,
                 prefetch_to_gpu=False,
                 prefetch_device=0):

        super(ToFeedDict, self).__init__()
        self.feed_vars = feed_vars
        self.extra_vars = extra_vars

        if prefetch_to_gpu:
            place = fluid.CUDAPlace(prefetch_device)
        elif pin_memory:
            place = fluid.CUDAPinnedPlace()
        else:
            place = fluid.CPUPlace()
        self.place = place

        self._normalized_vars = []
        for var in self.feed_vars:
            if isinstance(var, str):
                name = var
                fields = [var]
                lod_level = 0
            else:
                assert isinstance(var, Mapping), \
                    "feed_var should be either string or dict like object"
                name = var['name']
                if 'fields' in var:
                    fields = var['fields']
                else:
                    fields = [name]
                lod_level = 'lod_level' in var and var['lod_level'] or 0
            self._normalized_vars.append({
                'name': name,
                'fields': fields,
                'lod_level': lod_level})

    def __call__(self, batch):
        feed_dict = {}
        for var in self._normalized_vars:
            name = var['name']
            lod_level = var['lod_level']
            fields = var['fields']

            arr_list = []
            seq_length = None

            for idx, f in enumerate(fields):
                # XXX basically just for `im_shape`
                if isinstance(f, numbers.Number):
                    arr = f
                else:
                    arr = batch[f]

                if lod_level == 0:
                    # 'image' may already be stacked by `PadToStride`
                    if not isinstance(arr, np.ndarray):
                        arr = np.stack(arr)
                    arr_list.append(arr)
                    continue

                flat, seq_length = self._flatten(arr, lod_level + 1)
                arr_list.append(flat)

            # combine fields
            if len(fields) == 1:
                ndarray = arr_list[0]
            else:
                ndarray = np.column_stack(np.broadcast_arrays(*arr_list))
            feed_dict[name] = self._to_tensor(ndarray, seq_length)

        extra_dict = {key: batch[key] for key in self.extra_vars}
        return feed_dict, extra_dict

    def _to_tensor(self, ndarray, seq_length=None):
        t = fluid.core.LoDTensor()
        if seq_length is not None:
            t.set_recursive_sequence_lengths(seq_length)
        t.set(ndarray, self.place)
        return t

    def _flatten(self, arr, lod_level):
        flat = []
        seq_length = [[] for _ in range(lod_level)]

        def _recurse(data, result, level):
            if level == 0:
                flat.append(data)
                return
            result[0].append(len(data))
            for item in data:
                _recurse(item, result[1:], level - 1)

        _recurse(arr, seq_length, lod_level)
        return flat, seq_length


class DataLoaderBuilder(dataloader.DataLoader):
    """
    Constructs the dataloader.

    Args:
        dataset (object): dataset instance or dict.
        sampler (object): sampler instance or dict.
        batch_size (int): batch size.
        sample_transforms (list): list of data transformations to be performed
            on each sample.
        batch_transforms (list):  list of data transformations to be performed
            on each batch, after all samples are collected.
        feed_vars (list):  list of sample fields to be fed to the network
        extra_vars (list): list of sample fields to be used out of the network,
            e.g., for computing evaluation metrics
        num_workers (int): number of dataloader workers.
        multiprocessing (bool): use threading or multiprocessing.
        queue_depth (int): number of batches to buffer.
        pin_memory (bool): prefetch data to CUDA pinned memory.
        prefetch_to_gpu (bool): prefetch data to CUDA device.
    """
    __category__ = 'data'

    def __init__(self,
                 dataset,
                 sampler=None,
                 batch_size=1,
                 sample_transforms=[],
                 batch_transforms=[],
                 feed_vars=[],
                 extra_vars=[],
                 num_workers=0,
                 multiprocessing=False,
                 queue_depth=2,
                 pin_memory=False,
                 prefetch_to_gpu=False):
        if isinstance(dataset, dict):
            kwargs = dataset
            cls = kwargs.pop('type')
            dataset = type_map[cls](**kwargs)

        rank = 0
        world_size = 1
        init_seed = random.randint(0, 1e5)
        env = os.environ
        if 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env:
            rank = int(env['PADDLE_TRAINER_ID'])
            world_size = int(env['PADDLE_TRAINERS_NUM'])
            init_seed = 42 * world_size

        if isinstance(sampler, dict):
            kwargs = sampler
            kwargs['rank'] = rank
            kwargs['world_size'] = world_size
            kwargs['init_seed'] = init_seed
            # XXX currently we only have one default sampler
            if 'type' not in kwargs:
                sampler = samplers.Sampler(dataset, batch_size, **kwargs)

        if 'FLAGS_selected_gpus' in env:
            prefetch_device = int(env['FLAGS_selected_gpus'])
        else:
            prefetch_device = 0
        to_feed = ToFeedDict(feed_vars, extra_vars, pin_memory,
                             prefetch_to_gpu, prefetch_device)

        super(DataLoaderBuilder, self).__init__(
            dataset, sampler, sample_transforms, batch_transforms,
            to_feed, num_workers, multiprocessing, queue_depth, rank)

    def __iter__(self):
        _iter = super(DataLoaderBuilder, self).__iter__()

        def forever():
            while True:
                try:
                    yield next(_iter)
                except StopIteration:
                    self.reset()
        return forever()


@register
@serializable
class TrainDataLoader(DataLoaderBuilder):
    __doc__ = DataLoaderBuilder.__doc__


@register
@serializable
class EvalDataLoader(DataLoaderBuilder):
    __doc__ = DataLoaderBuilder.__doc__


@register
@serializable
class TestDataLoader(DataLoaderBuilder):
    __doc__ = DataLoaderBuilder.__doc__
