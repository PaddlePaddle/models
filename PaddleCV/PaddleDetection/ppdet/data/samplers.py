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

from __future__ import division
from __future__ import print_function

import math
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import numpy as np

__all__ = ['Sampler']


class Sampler(object):
    def __init__(self,
                 shuffle=True,
                 aspect_ratio_thresholds=None,
                 init_seed=1,
                 sync_seed_schedule=True):
        super(Sampler, self).__init__()
        assert not aspect_ratio_thresholds or \
            isinstance(aspect_ratio_thresholds, Sequence), \
            "if given, aspect_ratio_thresholds must be a sequence"
        self.shuffle = shuffle
        self.aspect_ratio_thresholds = aspect_ratio_thresholds
        self.init_seed = init_seed
        self.sync_seed_schedule = sync_seed_schedule
        self.epoch = 0
        self._step = 0

    def setup(self):
        for name in ['dataset', 'batch_size', 'rank', 'world_size']:
            assert hasattr(self, name), name + " must be set"
        whole_batch_size = self.world_size * self.batch_size
        if self.world_size > 1 and not self.sync_seed_schedule:
            print("`sync_seed_schedule` is recommended for distributed "
                  + "training, you may want to reconsider")
        if self.aspect_ratio_thresholds is None:
            self.num_batches = math.ceil(len(self.dataset) / whole_batch_size)
            return self

        assert hasattr(self.dataset, 'aspect_ratios'), \
            "aspect_ratio_thresholds is set, " + \
            "but dataset does not provide aspect ratio info"

        self.bucket_flags = np.digitize(self.dataset.aspect_ratios,
                                        self.aspect_ratio_thresholds)
        self.bucket_sizes = np.bincount(self.bucket_flags)
        self.pad_lengths = [
            int(math.ceil(s / whole_batch_size) * whole_batch_size) - s
            for s in self.bucket_sizes]
        self.num_batches = sum([int(math.ceil(s / whole_batch_size))
                                for s in self.bucket_sizes])

    def reset(self):
        if not self.shuffle:
            def rand_perm(x):
                return x
        elif self.sync_seed_schedule:
            print("set rank {} random seed to: {}".format(
                self.rank, self.epoch + self.init_seed))
            rand_perm = np.random.RandomState(self.epoch).permutation
            # XXX do not use with `itertools.cycle`,
            # should work fine for regular for loops or enumerate()
            self.epoch += 1
        else:
            rand_perm = np.random.permutation

        whole_batch_size = self.world_size * self.batch_size
        if self.aspect_ratio_thresholds:
            shuffled_indices = []
            for idx, (size, pad) in enumerate(
                    zip(self.bucket_sizes, self.pad_lengths)):
                if size == 0:
                    continue
                bucket = np.where(self.bucket_flags == idx)[0]
                shuffled = list(rand_perm(bucket))
                shuffled += shuffled[:pad]
                shuffled_indices += shuffled
        else:
            pad = len(self) * whole_batch_size - len(self.dataset)
            shuffled_indices = list(rand_perm(np.arange(len(self.dataset))))
            shuffled_indices += shuffled_indices[:pad]

        # shuffle by small batch, i.e., draw each small batch from same bucket
        shape = [-1, self.batch_size]

        # shuffle along batch index then split by number of shards
        batches = np.array(shuffled_indices).reshape(*shape)
        shuffled_shards = rand_perm(batches).reshape(
            self.world_size, -1, self.batch_size).tolist()
        self._shard = shuffled_shards[self.rank]
        self._step = 0

    def __iter__(self):
        if not hasattr(self, 'num_batches'):
            self.setup()
        if not hasattr(self, '_shard'):
            self.reset()
        return self

    def __next__(self):
        if self._step >= self.num_batches:
            raise StopIteration
        ids = self._shard[self._step]
        self._step += 1
        return ids

    def __len__(self):
        if not hasattr(self, 'num_batches'):
            self.setup()
        return self.num_batches
