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
Sampler class.
"""

import numpy as np


class Sampler(object):
    
    def __init__(self):
        return

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset
        return

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(range(len(self)))


class RandomSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0
        return

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        np.random.seed(self.epoch)
        self.epoch += 1
        return iter(np.random.permutation(len(self)))


class SortedSampler(Sampler):
    """ Sorted Sampler.
    
    Sort each block of examples by key.
    """

    def __init__(self, sampler, sort_pool_size, key="src"):
        self.sampler = sampler
        self.sort_pool_size = sort_pool_size
        self.key = lambda idx: len(self.sampler.dataset[idx][key])
        return

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        pool = []
        for idx in self.sampler:
            pool.append(idx)
            if len(pool) == self.sort_pool_size:
                pool = sorted(pool, key=self.key)
                for i in pool:
                    yield i
                pool = []
        if len(pool) > 0:
            pool = sorted(pool, key=self.key)
            for i in pool:
                yield i
