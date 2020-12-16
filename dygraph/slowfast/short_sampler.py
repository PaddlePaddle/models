from __future__ import print_function
from __future__ import division

import numpy as np
import math

from paddle.io import BatchSampler

__all__ = ["DistributedShortSampler"]


class DistributedShortSampler(BatchSampler):
    """Sampler that restricts data loading to a subset of the dataset.
    In such case, each process can pass a DistributedBatchSampler instance
    as a DataLoader sampler, and load a subset of the original dataset that
    is exclusive to it.
    .. note::
        Batch size is dynamic changed following short cycle schedule.

    Args:
        dataset(paddle.io.Dataset): this could be a `paddle.io.Dataset` implement
                     or other python object which implemented
                     `__len__` for BatchSampler to get sample
                     number of data source.
        batch_sizes(list): batch size list of one cycle.
        num_replicas(int, optional): porcess number in distributed training.
            If :attr:`num_replicas` is None, :attr:`num_replicas` will be
            retrieved from :code:`paddle.fluid.dygraph.parallel.ParallenEnv`.
            Default None.
        rank(int, optional): the rank of the current process among :attr:`num_replicas`
            processes. If :attr:`rank` is None, :attr:`rank` is retrieved from
            :code:`paddle.fluid.dygraph.parallel.ParallenEnv`. Default None.
        shuffle(bool): whther to shuffle indices order before genrating
            batch indices. Default False.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size. Default False
    """

    def __init__(self,
                 dataset,
                 batch_sizes,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False):
        self.dataset = dataset

        assert any(isinstance(batch_size, int) and batch_size > 0 for batch_size in batch_sizes), \
            "batch_size should be a positive integer"
        self.batch_sizes = batch_sizes
        self.len_batch_sizes = len(self.batch_sizes)
        assert isinstance(shuffle, bool), \
            "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
            "drop_last should be a boolean number"

        from paddle.distributed import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices)
                             )]  #completion last iter
        assert len(indices) == self.total_size
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            total_batch_size = sum(self.batch_sizes)
            subsampled_indices = []
            last_batch_size = self.total_size % (
                total_batch_size * self.nranks)  #number samples of last batch
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * total_batch_size,
                           len(indices) - last_batch_size,
                           total_batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + total_batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples  #index length in each card
        _sample_iter = iter(indices)

        batch_indices = []
        counter = 0
        batch_size = self.batch_sizes[0]
        for idx in _sample_iter:
            batch_indices.append(
                (idx, counter %
                 self.len_batch_sizes))  #to be used in dataloader get_item
            if len(batch_indices) == batch_size:
                yield batch_indices
                counter += 1
                batch_size = self.batch_sizes[counter % self.len_batch_sizes]
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        avg_batch_size = sum(self.batch_sizes) / float(self.len_batch_sizes)
        if self.drop_last:
            return int(np.floor(self.num_samples / avg_batch_size))
        else:
            return int(np.ceil(self.num_samples / avg_batch_size))

    def set_epoch(self, epoch):
        """
        Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.
        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
