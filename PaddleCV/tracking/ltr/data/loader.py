import os
import signal
import sys

import dataflow as df
import numpy as np


# handle terminate reader process, do not print stack frame
def _reader_quit(signum, frame):
    print("Reader process exit.")
    sys.exit()


def _term_group(sig_num, frame):
    print('pid {} terminated, terminate group '
          '{}...'.format(os.getpid(), os.getpgrp()))
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


signal.signal(signal.SIGTERM, _reader_quit)
signal.signal(signal.SIGINT, _term_group)


class LTRLoader(df.DataFlow):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Note: an additional option stack_dim is available to
            select along which dimension the data should be stacked to form a batch.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        stack_dim (int): Dimension along which to stack to form the batch. (default: 0)
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)


    .. warning:: If ``spawn`` start method is used, :attr:`worker_init_fn` cannot be an
                 unpicklable object, e.g., a lambda function.
    """

    __initialized = False

    def __init__(self,
                 name,
                 dataset,
                 training=True,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 epoch_interval=1,
                 collate_fn=None,
                 stack_dim=0,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None):

        super().__init__()

        ds = df.RepeatedData(dataset, -1)
        ds = df.MultiProcessRunnerZMQ(ds, num_proc=num_workers, hwm=300)
        # ds = df.MultiThreadRunner(lambda: ds, num_prefetch=1024, num_thread=num_workers)
        ds = df.BatchData(ds, batch_size)
        self.ds = ds

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim
        self.batches_per_epoch = len(dataset) // batch_size

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        if not self.__initialized:
            self.reset_state()
            self.__initialized = True

        for d in self.ds:
            if self.stack_dim > 0:
                for k, v in d.items():
                    if len(v.shape) >= self.stack_dim + 1:
                        d[k] = np.swapaxes(v, 0, self.stack_dim)
            yield d

    def reset_state(self):
        self.ds.reset_state()
