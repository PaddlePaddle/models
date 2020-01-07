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

import glob
import six
import os
import tarfile
import random

import numpy as np


from collections import defaultdict

def batching_scheme(batch_size,
                    max_length,
                    min_length_bucket=8,
                    length_bucket_step=1.1,
                    drop_long_sequences=False,
                    shard_multiplier=1,
                    length_multiplier=1,
                    min_length=0):
    """A batching scheme based on model hyperparameters.

    Every batch containins a number of sequences divisible by `shard_multiplier`.

    Args:
      batch_size: int, total number of tokens in a batch.
      max_length: int, sequences longer than this will be skipped. Defaults to
        batch_size.
      min_length_bucket: int
      length_bucket_step: float greater than 1.0
      drop_long_sequences: bool, if True, then sequences longer than
        `max_length` are dropped.  This prevents generating batches with
        more than the usual number of tokens, which can cause out-of-memory
        errors.
      shard_multiplier: an integer increasing the batch_size to suit splitting
        across datashards.
      length_multiplier: an integer multiplier that is used to increase the
        batch sizes and sequence length tolerance.
      min_length: int, sequences shorter than this will be skipped.

    Returns:
       A dictionary with parameters that can be passed to input_pipeline:
         * boundaries: list of bucket boundaries
         * batch_sizes: list of batch sizes for each length bucket
         * max_length: int, maximum length of an example

    Raises:
      ValueError: If min_length > max_length
    """

    def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
        assert length_bucket_step > 1.0
        x = min_length
        boundaries = []
        while x < max_length:
            boundaries.append(x)
            x = max(x + 1, int(x * length_bucket_step))
        return boundaries

    max_length = max_length or batch_size
    if max_length < min_length:
        raise ValueError("max_length must be greater or equal to min_length")

    boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                    length_bucket_step)
    boundaries = [boundary * length_multiplier for boundary in boundaries]
    max_length *= length_multiplier

    batch_sizes = [
        max(1, batch_size // length) for length in boundaries + [max_length]
        ]
    max_batch_size = max(batch_sizes)
    # Since the Datasets API only allows a single constant for window_size,
    # and it needs divide all bucket_batch_sizes, we pick a highly-compoisite
    # window size and then round down all batch sizes to divisors of that window
    # size, so that a window can always be divided evenly into batches.
    # TODO(noam): remove this when Dataset API improves.
    highly_composite_numbers = [
        1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
        2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
        83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
        720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
        7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
        36756720, 43243200, 61261200, 73513440, 110270160
    ]
    window_size = max(
        [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
    divisors = [i for i in xrange(1, window_size + 1) if window_size % i == 0]
    batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
    window_size *= shard_multiplier
    batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
    # The Datasets API splits one window into multiple batches, which
    # produces runs of many consecutive batches of the same size.  This
    # is bad for training.  To solve this, we will shuffle the batches
    # using a queue which must be several times as large as the maximum
    # number of batches per window.
    max_batches_per_window = window_size // min(batch_sizes)
    shuffle_queue_size = max_batches_per_window * 3

    ret = {
        "boundaries": boundaries,
        "batch_sizes": batch_sizes,
        "min_length": min_length,
        "max_length": (max_length if drop_long_sequences else 10 ** 9),
        "shuffle_queue_size": shuffle_queue_size,
    }
    return ret


def bucket_by_sequence_length(data_reader,
                              example_length_fn,
                              bucket_boundaries,
                              bucket_batch_sizes, 
                              trainer_nums,
                              trainer_id):
    """Bucket entries in dataset by length.

    Args:
      dataset: Dataset of dict<feature name, Tensor>.
      example_length_fn: function from example to int, determines the length of
        the example, which will determine the bucket it goes into.
      bucket_boundaries: list<int>, boundaries of the buckets.
      bucket_batch_sizes: list<int>, batch size per bucket.

    Returns:
      Dataset of padded and batched examples.
    """
    def example_to_bucket_id(example):
        """
            get bucket_id
        """
        seq_length = example_length_fn(example)
        boundaries = list(bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        for i in range(len(buckets_min)):
            if buckets_min[i] <= seq_length and seq_length < buckets_max[i]:
                bucket_id = i
        return bucket_id

    def window_size_fn(bucket_id):
        """
            get window size
        """
        window_size = bucket_batch_sizes[bucket_id]
        return window_size

    def group_by_window(reader, key_func, window_size_func, drop_last=False):
        """
            group the line by length
        """
        groups = defaultdict(list)

        def impl():
            """
                impl
            """
            for e in reader():
                key = key_func(e)
                window_size = window_size_func(key)
                groups[key].append(e)
                if len(groups[key]) == window_size:
                    each_size = window_size / trainer_nums
                    res = groups[key][trainer_id * each_size: (trainer_id + 1) * each_size]
                    yield res
                    groups[key] = []
            if drop_last:
                groups.clear()

        return impl

    reader = group_by_window(data_reader, example_to_bucket_id, window_size_fn)
    return reader


def shuffle(reader, buf_size):
    """
    Creates a data reader whose data output is shuffled.

    Output from the iterator that created by original reader will be
    buffered into shuffle buffer, and then shuffled. The size of shuffle buffer
    is determined by argument buf_size.

    :param reader: the original reader whose output will be shuffled.
    :type reader: callable
    :param buf_size: shuffle buffer size.
    :type buf_size: int

    :return: the new reader whose output is shuffled.
    :rtype: callable
    """

    def data_reader():
        """
            data_reader
        """
        buf = []
        for e in reader():
            buf.append(e)
            if len(buf) >= buf_size:
                random.shuffle(buf)
                for b in buf:
                    yield b
                buf = []

        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b

    return data_reader


def sort(reader, buf_size, cmp=None, key=None, reverse=False):
    """
    Creates a data reader whose data output is sorted.

    Output from the iterator that created by original reader will be
    buffered into sort buffer, and then sorted. The size of sort buffer
    is determined by argument buf_size.

    :param reader: the original reader whose output will be sorted.
    :type reader: callable
    :param buf_size: shuffle buffer size.
    :type buf_size: int

    :return: the new reader whose output is sorted.
    :rtype: callable
    """

    def data_reader():
        """
            data_reader
        """
        buf = []
        for e in reader():
            buf.append(e)
            if len(buf) >= buf_size:
                buf = sorted(buf, cmp, key, reverse)
                for b in buf:
                    yield b
                buf = []

        if len(buf) > 0:
            sorted(buf, cmp, key, reverse)
            for b in buf:
                yield b

    return data_reader


def batch_by_token(reader, batch_size, len_fun, drop_last=False):
    """
    Create a batched reader.

    :param reader: the data reader to read from.
    :type reader: callable
    :param batch_size: size of each mini-batch
    :type batch_size: int
    :param drop_last: drop the last batch, if the size of last batch is not equal to batch_size.
    :type drop_last: bool
    :return: the batched reader.
    :rtype: callable
    """

    def batch_reader():
        """
            batch_reader
        """
        r = reader()
        b = []
        max_len = 0
        for instance in r:
            cur_len = len_fun(instance)
            max_len = max(max_len, cur_len)
            if max_len * (len(b) + 1) > batch_size:
                yield b
                b = [instance]
                max_len = cur_len
            else:
                b.append(instance)
        if drop_last == False and len(b) != 0:
            yield b

    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size should be a positive integeral value, "
                         "but got batch_size={}".format(batch_size))

    return batch_reader


def parse_line(line, max_len, min_len=0, field_delimiter="\t", token_delimiter=" "):
    """
        parse training data
    """
    src, trg = line.strip("\n").split(field_delimiter)
    src_ids = [int(token) for token in src.split(token_delimiter)]
    trg_ids = [int(token) for token in trg.split(token_delimiter)]
    reverse_trg_ids = trg_ids[::-1]
    reverse_trg_ids = reverse_trg_ids[1:]
    reverse_trg_ids.append(1)
    inst_max_len = max(len(src_ids), len(trg_ids))
    inst_min_len = min(len(src_ids), len(trg_ids))
    if inst_max_len <= max_len and inst_min_len > min_len:
        return src_ids, [0] + trg_ids[:-1], trg_ids, [0] + reverse_trg_ids[:-1], reverse_trg_ids
    else:
        return None


def repeat(reader, count=-1):
    """
        repeat
    """
    def data_reader():
        """
            repeat data
        """
        time = count
        while time != 0:
            for e in reader():
                yield e
            time -= 1

    return data_reader


def parse_src_line(line, max_len, min_len=0, token_delimiter=" "):
    """
        parse infer data
    """
    src = line.strip("\n")
    src_ids = [int(token) for token in src.split(token_delimiter)]
    inst_max_len = inst_min_len = len(src_ids)
    if inst_max_len < max_len and inst_min_len > min_len:
        src_ids.append(1)
        return [src_ids]
    else:
        src_ids = src_ids[:max_len - 1]
        src_ids.append(1)
        return [src_ids]


def interleave_reader(fpattern, cycle_length, block_length=1, **kwargs):
    """
        cycle reader
    """
    # refer to:
    # https://www.tensorflow.org/api_docs/python/tf/contrib/data/parallel_interleave?hl=zh_cn
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset?hl=zh_cn#interleave
    fpaths = glob.glob(fpattern)
    fpaths = sorted(fpaths)
    if 'parse_line' in kwargs:

        parse_line = kwargs.pop('parse_line')

    class Worker(object):  # mimic a worker thread
        """
           each worker wrap a file
        """
        def __init__(self):
            self.input = None
            self.iter = None

        def set_input(self, input_arg):
            """
                set file reader
            """
            if self.iter is not None:
                self.iter.close()
            self.input = input_arg
            self.iter = open(input_arg, 'rb')

        def get_next(self):
            """
                get next data
            """
            return next(self.iter)

    def data_reader():
        """
            generate data
        """
        num_workers = cycle_length  # + prefetched
        workers = []
        # Indices in `workers` of iterators to interleave.
        interleave_indices = []
        # Indices in `workers` of prefetched iterators.
        staging_indices = []
        # EnsureWorkerThreadsStarted
        for i in range(num_workers):
            if i >= len(fpaths):
                break
            workers.append(Worker())
            workers[i].set_input(fpaths[i])
            if i < cycle_length:
                interleave_indices.append(i)
            else:
                staging_indices.append(i)
        input_index = len(workers)  # index for files
        next_index = 0  # index for worker
        block_count = 0  # counter for the number of instances from one block
        #
        while True:  # break while when all inputs end
            can_produce_elements = False
            # The for loop only fully runs when all workers ending.
            # Otherwise, run one step then break the for loop, or
            # find the first possible unended iterator by setting next_index
            # or go to the step of loop.
            for i in range(len(interleave_indices)):
                index = (next_index + i) % len(interleave_indices)
                current_worker_index = interleave_indices[index]
                current_worker = workers[current_worker_index]

                try:
                    line = current_worker.get_next()
                    if six.PY3:
                        line = line.decode()
                    inst = parse_line(line, **kwargs)
                    if inst is not None:
                        yield inst
                    next_index = index
                    block_count += 1
                    if block_count == block_length:
                        # advance to the next iterator
                        next_index = (index + 1) % len(interleave_indices)
                        block_count = 0
                    can_produce_elements = True
                    break
                except (StopIteration,):  # This iterator has reached the end.
                    if input_index < len(fpaths):  # get a new iterator and skip
                        current_worker.set_input(fpaths[input_index])
                        staging_indices.append(current_worker_index)
                        if len(staging_indices) > 0:  # pop_front
                            interleave_indices[index] = staging_indices[0]
                            staging_indices = staging_indices[1:]

                        input_index += 1
                        # advance to the next iterator
                        next_index = (index + 1) % len(interleave_indices)
                        block_count = 0
                        can_produce_elements = True
                        break
                    # else: advance to the next iterator by loop step

            if not can_produce_elements:
                # all inputs end, triggered when all iterators have reached the end
                break

    return data_reader


def line_reader(fpattern, batch_size, dev_count, **kwargs):
    """
        cycle reader
    """

    fpaths = glob.glob(fpattern)
    #np.random.shuffle(fpaths)
    #random.shuffle(fpaths) 
    if "parse_line" in kwargs:
        parse_line = kwargs.pop('parse_line')

    def data_reader():
        """
            data_reader
        """
        res = []
        total_size = batch_size * dev_count
        for fpath in fpaths:
            if not os.path.isfile(fpath):
                raise IOError("Invalid file: %s" % fpath)
            with open(fpath, "rb") as f:
                for line in f:
                    if six.PY3:
                        line = line.decode()
                    inst = parse_line(line, **kwargs)
                    res.append(inst)
                    if len(res) == total_size:
                        yield res
                        res = []
        if len(res) > 0:
            pad_count = total_size - len(res)
            for index in xrange(pad_count):
                res.append(res[-1])
            yield res

    return data_reader


def prepare_data_generator(args, is_test, count, pyreader, batch_size=None, 
                            data_reader=None, py_reader_provider_wrapper=None):
    """
    Data generator wrapper for DataReader. If use py_reader, set the data
    provider for py_reader
    """
    def stack(data_reader, count, clip_last=True):
        """
            Data generator for multi-devices
        """
        def __impl__():
            res = []
            for item in data_reader():
                res.append(item)
                if len(res) == count:
                    yield res
                    res = []
            if len(res) == count:
                yield res
            elif not clip_last:
                data = []
                for item in res:
                    data += item
                if len(data) > count:
                    inst_num_per_part = len(data) // count
                    yield [
                        data[inst_num_per_part * i:inst_num_per_part * (i + 1)]
                        for i in range(count)
                    ]

        return __impl__

    def split(data_reader, count):
        """
            split for multi-gpu
        """
        def __impl__():
            for item in data_reader():
                inst_num_per_part = len(item) // count
                for i in range(count):
                    yield item[inst_num_per_part * i:inst_num_per_part * (i + 1
                                                                          )]

        return __impl__

    if not args.use_token_batch:
        # to make data on each device have similar token number
        data_reader = split(data_reader, count)
    #if args.use_py_reader:
    if pyreader:
        pyreader.decorate_tensor_provider(
            py_reader_provider_wrapper(data_reader))
        data_reader = None
    else:  # Data generator for multi-devices
        data_reader = stack(data_reader, count)
    return data_reader


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array(
            [[1.] * len(inst) + [0.] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            list(range(0, len(inst))) + [0] * (max_len - len(inst))
            for inst in insts
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]
