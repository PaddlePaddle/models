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

import os
import random

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


class DataLoaderBuilder(dataloader.DataLoader):
    """
    DataLoader for loading data

    Args:
        dataset (object): dataset instance or dict.
        sampler (object): sampler instance or dict.
        batch_size (int): batch size.
        sample_transforms (list): list of data transformations to be performed
            on each sample.
        batch_transforms (list):  list of data transformations to be performed
            on each batch, after all samples are collected.
        num_workers (int): number of dataloader workers.
        multiprocessing (bool): use threading or multiprocessing.
        read_ahead (int): number of batches to read ahead.
    """
    __category__ = 'data'

    def __init__(self,
                 dataset,
                 sampler=None,
                 batch_size=1,
                 sample_transforms=[],
                 batch_transforms=[],
                 num_workers=0,
                 multiprocessing=False,
                 read_ahead=2):
        if isinstance(sampler, dict):
            if 'type' not in sampler:
                sampler = samplers.Sampler(**sampler)

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

        super(DataLoaderBuilder, self).__init__(
            dataset, sampler, batch_size, sample_transforms, batch_transforms,
            num_workers, multiprocessing, read_ahead,
            init_seed, rank, world_size)

    def __iter__(self):
        _iter = super(DataLoaderBuilder, self).__iter__()

        def forever():
            while True:
                try:
                    yield next(_iter)
                except StopIteration:
                    self.reset()
        return forever


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
