#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
DataLoader class
"""

import math

import paddle.fluid as fluid
import paddle.batch

from plato.args import str2bool
from plato.data.sampler import RandomSampler
from plato.data.sampler import SequentialSampler
from plato.data.sampler import SortedSampler
import plato.modules.parallel as parallel


class DataLoader(object):
    """ Implement of DataLoader. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument("--shuffle", type=str2bool, default=True)
        group.add_argument("--sort_pool_size", type=int, default=0)
        return group

    def __init__(self, dataset, hparams, collate_fn=None, sampler=None, is_test=False, is_train=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.sort_pool_size = hparams.sort_pool_size

        if sampler is None:
            if hparams.shuffle and not is_test:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if self.sort_pool_size > 0 and not is_test:
            sampler = SortedSampler(sampler, self.sort_pool_size)

        def reader():
            for idx in sampler:
                yield idx

        self.reader = paddle.batch(reader, batch_size=hparams.batch_size, drop_last=False)
        self.num_batches = math.ceil(len(dataset) / hparams.batch_size)

        if hparams.use_data_distributed and parallel.Env().nranks > 1 and is_train:
            self.reader = fluid.contrib.reader.distributed_batch_reader(self.reader)
            self.num_batches = self.num_batches // fluid.dygraph.parallel.Env().nranks

        return

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for batch_indices in self.reader():
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)
