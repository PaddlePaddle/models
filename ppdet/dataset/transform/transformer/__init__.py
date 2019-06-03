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

from . import base
from .parallel_map import ParallelMappedDataset


def map(ds, mapper, worker_args=None):
    """ apply 'mapper' to 'ds'

    Args:
        ds (instance of Dataset): dataset to be mapped
        mapper (function): action to be executed for every data sample
        worker_args (dict): configs for concurrent mapper

    Returns:
        a mapped dataset
    """
    if worker_args is not None:
        return ParallelMappedDataset(ds, mapper, worker_args)
    else:
        return base.MappedDataset(ds, mapper)


def batch(ds, batchsize, coarsest_stride, drop_last=True, is_padding=False, 
          random_shapes=[], multi_scales=[]):
    """ Batch data samples to batches

    Args:
        batchsize (int): number of samples for a batch
        coarsest_stride(int): stride of the coarsest FPN level
        drop_last (bool): drop last few samples if not enough for a batch
        is_padding (bool): whether padding the image in one batch
        random_shapes: (list of int): resize to image to random 
                                      shapes, [] for not resize.
        random_shapes: (list of int): resize image by random 
                                      scales, [] for not resize.

    Returns:
        a batched dataset
    """
    return base.BatchedDataset(
        ds,
        batchsize,
        coarsest_stride,
        drop_last=drop_last,
        is_padding=is_padding,
        random_shapes=random_shapes,
        multi_scales=multi_scales)


__all__ = ['map', 'batch']
