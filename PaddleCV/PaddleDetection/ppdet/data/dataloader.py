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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np

__all__ = ['DataLoader']


class _Compose(object):
    def __init__(self, transforms):
        super(_Compose, self).__init__()
        assert transforms and isinstance(transforms, Sequence), \
            "sample_transforms must a sequence of callables"
        self.transforms = transforms
        mixup = list(filter(lambda x: hasattr(x, 'is_mixup'), transforms))
        self.use_mixup = bool(mixup)
        if self.use_mixup:
            self.mixup_steps = mixup[0].steps

    @property
    def need_seeding(self):
        if not hasattr(self, 'batch_seed'):
            self.batch_seed = bool(list(filter(
                lambda x: hasattr(x, 'batch_seed'), self.transforms)))
        return self.batch_seed

    def __call__(self, sample):
        if isinstance(sample, tuple):
            sample, sample2 = sample

        for transform in self.transforms:
            if hasattr(transform, 'is_mixup'):
                sample = transform(sample, sample2)
            else:
                sample = transform(sample)

        return sample


class _Batchify(object):
    def __init__(self, transforms=[]):
        super(_Batchify, self).__init__()
        self.transforms = transforms

    def __call__(self, batch):
        # array of structures to structure of arrays
        batch_dict = {}
        for sample in batch:
            for k, v in sample.items():
                if k not in batch_dict:
                    batch_dict[k] = []
                batch_dict[k].append(v)

        for transform in self.transforms:
            batch_dict = transform(batch_dict)
        return batch_dict


def _apply_transform(idx, dataset, transform, batch_seed=None, step=0):
    assert not transform.use_mixup or dataset.mode == 'indexable', \
        'mixup only works with indexable datasets'

    if dataset.mode == 'iterable':
        return transform(next(dataset))

    sample = dataset[idx]
    # for random shape, ensure same size is chosen for the same batch
    # batch_seed = batch_idx * rank
    if batch_seed is not None:
        sample['batch_seed'] = batch_seed
    if transform.use_mixup and (transform.mixup_steps > step
                                or transform.mixup_steps < 0):
        idx2 = np.random.choice(np.delete(np.arange(
            len(dataset)), idx))
        if hasattr(dataset, 'samples'):
            # XXX sample2 is read-only, avoid deepcopy if possible
            sample2 = dataset.samples[idx2]
            sample2['image'] = dataset._read_image(sample2['file'])
        else:
            sample2 = dataset[idx2]
        sample = (sample, sample2)
    return transform(sample)


def _process_worker_init(dataset, transforms, batchify):
    global _worker_context
    _worker_context = (dataset, transforms, batchify)


def _process_worker_fn(idx, ids, context=None, batch_seed=None):
    global _worker_context
    dataset, transform, batchify = _worker_context
    samples = [_apply_transform(i, dataset, transform, batch_seed, idx)
               for i in ids]
    return batchify(samples)


def _thread_worker_fn(idx, ids, context, batch_seed=None):
    dataset, transform, batchify = context
    samples = [_apply_transform(i, dataset, transform, batch_seed, idx)
               for i in ids]
    return batchify(samples)


class _SingleWorkerLoaderIter(object):
    def __init__(self, loader):
        super(_SingleWorkerLoaderIter, self).__init__()
        self.dataset = loader.dataset
        self.transform = loader.transform
        self.batchify = loader.batchify
        self.rank = loader.rank
        self._iter = iter(loader.sampler)
        self._batch_idx = loader.step

    def _batch_seed(self):
        if self.transform.need_seeding:
            return (self._batch_idx + 1) * (self.rank + 1)
        else:
            return None

    def __next__(self):
        ids = next(self._iter)
        samples = [
            _apply_transform(i, self.dataset, self.transform,
                             self._batch_seed(), self._batch_idx) for i in ids]
        batch = self.batchify(samples)
        self._batch_idx += 1
        return batch

    next = __next__


class _MultiWorkerLoaderIter(object):
    def __init__(self, loader):
        super(_MultiWorkerLoaderIter, self).__init__()
        self.dataset = loader.dataset
        self.transform = loader.transform
        self.rank = loader.rank
        self.buffer_size = loader.buffer_size
        self._iter = iter(loader.sampler)
        self._out_buffer = {}
        self._recv_idx = loader.step
        self._sent_idx = loader.step

        worker_context = (loader.dataset, loader.transform, loader.batchify)
        if loader.multiprocessing:
            # `maxtasksperchild` is needed to avoid OOM
            self._worker_pool = Pool(
                loader.num_workers, initializer=_process_worker_init,
                initargs=worker_context, maxtasksperchild=4 * self.buffer_size)
            self._worker_fn = _process_worker_fn
            self._worker_context = None
        else:
            self._worker_pool = ThreadPool(loader.num_workers)
            self._worker_fn = _thread_worker_fn
            self._worker_context = worker_context

        for _ in range(loader.buffer_size):
            self._queue_next()

    def _batch_seed(self):
        if self.transform.need_seeding:
            return np.random.randint(0, 2**31)

    def _queue_next(self):
        if self._sent_idx - self._recv_idx > self.buffer_size:
            return
        ids = next(self._iter, None)
        if ids is None:
            return
        future = self._worker_pool.apply_async(
            self._worker_fn,
            (self._sent_idx, ids, self._worker_context, self._batch_seed()))
        self._out_buffer[self._sent_idx] = future
        self._sent_idx += 1

    def __next__(self):
        steps_ahead = len(self._out_buffer)
        for _ in range(self.buffer_size + 1 - steps_ahead):
            self._queue_next()
        if self._recv_idx == self._sent_idx:
            assert not self._out_buffer, "result queue should be empty by now"
            raise StopIteration
        assert self._recv_idx < self._sent_idx
        assert self._recv_idx in self._out_buffer
        future = self._out_buffer.pop(self._recv_idx)
        batch = future.get()
        self._recv_idx += 1
        return batch

    next = __next__


class DataLoader(object):
    def __init__(self,
                 dataset,
                 sampler=None,
                 sample_transforms=[],
                 batch_transforms=None,
                 num_workers=0,
                 multiprocessing=False,
                 buffer_size=2,
                 rank=0):
        super(DataLoader, self).__init__()

        self.dataset = dataset
        self.sampler = sampler
        self.sample_transforms = sample_transforms
        self.batch_transforms = batch_transforms
        self.num_workers = num_workers
        self.multiprocessing = multiprocessing
        self.buffer_size = buffer_size
        self.rank = rank
        self.step = 0

        assert sampler is not None or dataset.mode != 'indexable', \
            "Sampler is required for indexable dataset"

        self.transform = _Compose(sample_transforms)
        self.batchify = _Batchify(batch_transforms)

    def __len__(self):
        if self.dataset.mode == 'indexable':
            return len(self.sampler)
        raise TypeError('Iterable DataSet does not have fixed length')

    def __iter__(self):
        if self.num_workers > 0:
            return _MultiWorkerLoaderIter(self)
        else:
            return _SingleWorkerLoaderIter(self)

    def reset(self):
        if self.sampler is not None:
            self.sampler.reset()
