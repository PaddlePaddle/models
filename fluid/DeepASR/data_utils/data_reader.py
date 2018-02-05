"""This module contains data processing related logic.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import struct
import data_utils.augmentor.trans_mean_variance_norm as trans_mean_variance_norm
import data_utils.augmentor.trans_add_delta as trans_add_delta
from multiprocessing import Manager, Process
from threading import Thread
import time


class SampleInfo(object):
    """SampleInfo holds the necessary information to load an example from disk.

    Args:
        feature_bin_path (str): File containing the feature data.
        feature_start (int): Start position of the sample's feature data.
        feature_size (int): Byte count of the sample's feature data.
        feature_frame_num (int): Time length of the sample.
        feature_dim (int): Feature dimension of one frame.
        label_bin_path (str): File containing the label data.
        label_size (int): Byte count of the sample's label data. 
        label_frame_num (int): Label number of the sample.
    """

    def __init__(self, feature_bin_path, feature_start, feature_size,
                 feature_frame_num, feature_dim, label_bin_path, label_start,
                 label_size, label_frame_num):
        self.feature_bin_path = feature_bin_path
        self.feature_start = feature_start
        self.feature_size = feature_size
        self.feature_frame_num = feature_frame_num
        self.feature_dim = feature_dim

        self.label_bin_path = label_bin_path
        self.label_start = label_start
        self.label_size = label_size
        self.label_frame_num = label_frame_num


class SampleInfoBucket(object):
    """SampleInfoBucket contains paths of several description files. Feature
    description file contains necessary information to access samples' feature 
    data and label description file contains necessary information to
    access samples' label data. SampleInfoBucket is the minimum unit to do 
    shuffle.

    Args:
        feature_bin_paths (list|tuple): Files containing the binary feature 
                                        data.
        feature_desc_paths (list|tuple): Files containing the description of 
                                         samples' feature data. 
        label_bin_paths (list|tuple): Files containing the binary label data.
        label_desc_paths (list|tuple): Files containing the description of
                                       samples' label data.
    """

    def __init__(self, feature_bin_paths, feature_desc_paths, label_bin_paths,
                 label_desc_paths):
        block_num = len(label_bin_paths)
        assert len(label_desc_paths) == block_num
        assert len(feature_bin_paths) == block_num
        assert len(feature_desc_paths) == block_num
        self._block_num = block_num

        self._feature_bin_paths = feature_bin_paths
        self._feature_desc_paths = feature_desc_paths
        self._label_bin_paths = label_bin_paths
        self._label_desc_paths = label_desc_paths

    def generate_sample_info_list(self):
        sample_info_list = []
        for block_idx in xrange(self._block_num):
            label_bin_path = self._label_bin_paths[block_idx]
            label_desc_path = self._label_desc_paths[block_idx]
            feature_bin_path = self._feature_bin_paths[block_idx]
            feature_desc_path = self._feature_desc_paths[block_idx]

            label_desc_lines = open(label_desc_path).readlines()
            feature_desc_lines = open(feature_desc_path).readlines()

            sample_num = int(label_desc_lines[0].split()[1])
            assert sample_num == int(feature_desc_lines[0].split()[1])

            for i in xrange(sample_num):
                feature_desc_split = feature_desc_lines[i + 1].split()
                feature_start = int(feature_desc_split[2])
                feature_size = int(feature_desc_split[3])
                feature_frame_num = int(feature_desc_split[4])
                feature_dim = int(feature_desc_split[5])

                label_desc_split = label_desc_lines[i + 1].split()
                label_start = int(label_desc_split[2])
                label_size = int(label_desc_split[3])
                label_frame_num = int(label_desc_split[4])

                sample_info_list.append(
                    SampleInfo(feature_bin_path, feature_start, feature_size,
                               feature_frame_num, feature_dim, label_bin_path,
                               label_start, label_size, label_frame_num))

        return sample_info_list


class EpochEndSignal():
    pass


class DataReader(object):
    """DataReader provides basic audio sample preprocessing pipeline including
    I/O and augmentation transformation.

    Args:
        feature_file_list (str): File containing feature data related files.
        label_file_list (str): File containing label data related files.
        frame_dim (int): The final feature dimension of one frame after all 
                         augmentation applied.
        drop_sentence_len (int): Lower threshold bound to filter samples having 
                                 long sentence.
        process_num (int): Number of processes for processing data.
        sample_buffer_size (int): Buffer size to indicate the maximum samples 
                                  cached.
        sample_info_buffer_size (int): Buffer size to indicate the maximum 
                                       sample information cached.
        shuffle_block_num (int): Block number indicating the minimum unit to do 
                                 shuffle.
        random_seed (int): Random seed.
    """

    def __init__(
            self,
            feature_file_list,
            label_file_list,
            frame_dim=120 * 11,  # @TODO augmentor is responsible for the value
            drop_sentence_len=512,
            drop_frame_len=256,
            process_num=10,
            sample_buffer_size=1024,
            sample_info_buffer_size=10000,
            shuffle_block_num=1,
            random_seed=0):
        self._feature_file_list = feature_file_list
        self._label_file_list = label_file_list
        self._drop_sentence_len = drop_sentence_len
        self._frame_dim = frame_dim
        self._drop_frame_len = drop_frame_len
        self._shuffle_block_num = shuffle_block_num
        self._block_info_list = None
        self._rng = random.Random(random_seed)
        self._bucket_list = None
        self.generate_bucket_list(True)
        self._order_id = 0
        self._manager = Manager()
        self._sample_buffer_size = sample_buffer_size
        self._sample_info_buffer_size = sample_info_buffer_size
        self._process_num = process_num

    def generate_bucket_list(self, is_shuffle):
        if self._block_info_list is None:
            block_feature_info_lines = open(self._feature_file_list).readlines()
            block_label_info_lines = open(self._label_file_list).readlines()
            assert len(block_feature_info_lines) == len(block_label_info_lines)
            self._block_info_list = []
            for i in xrange(0, len(block_feature_info_lines), 2):
                block_info = (block_feature_info_lines[i],
                              block_feature_info_lines[i + 1],
                              block_label_info_lines[i],
                              block_label_info_lines[i + 1])
                self._block_info_list.append(
                    map(lambda line: line.strip(), block_info))

        if is_shuffle:
            self._rng.shuffle(self._block_info_list)

        self._bucket_list = []
        for i in xrange(0, len(self._block_info_list), self._shuffle_block_num):
            bucket_block_info = self._block_info_list[i:i +
                                                      self._shuffle_block_num]
            self._bucket_list.append(
                SampleInfoBucket(
                    map(lambda info: info[0], bucket_block_info),
                    map(lambda info: info[1], bucket_block_info),
                    map(lambda info: info[2], bucket_block_info),
                    map(lambda info: info[3], bucket_block_info)))

    # @TODO make this configurable
    def set_transformers(self, transformers):
        self._transformers = transformers

    def _sample_generator(self):
        sample_info_queue = self._manager.Queue(self._sample_info_buffer_size)
        sample_queue = self._manager.Queue(self._sample_buffer_size)
        self._order_id = 0

        def ordered_feeding_worker(sample_info_queue):
            for sample_info_bucket in self._bucket_list:
                sample_info_list = sample_info_bucket.generate_sample_info_list(
                )
                self._rng.shuffle(sample_info_list)  # do shuffle here
                for sample_info in sample_info_list:
                    sample_info_queue.put((sample_info, self._order_id))
                    self._order_id += 1

            for i in xrange(self._process_num):
                sample_info_queue.put(EpochEndSignal())

        feeding_thread = Thread(
            target=ordered_feeding_worker, args=(sample_info_queue, ))
        feeding_thread.daemon = True
        feeding_thread.start()

        def ordered_processing_worker(sample_info_queue, sample_queue,
                                      out_order):
            def read_bytes(fpath, start, size):
                f = open(fpath, 'r')
                f.seek(start, 0)
                binary_bytes = f.read(size)
                f.close()
                return binary_bytes

            ins = sample_info_queue.get()

            while not isinstance(ins, EpochEndSignal):
                sample_info, order_id = ins

                feature_bytes = read_bytes(sample_info.feature_bin_path,
                                           sample_info.feature_start,
                                           sample_info.feature_size)

                label_bytes = read_bytes(sample_info.label_bin_path,
                                         sample_info.label_start,
                                         sample_info.label_size)

                assert sample_info.label_frame_num * 4 == len(label_bytes)
                label_array = struct.unpack('I' * sample_info.label_frame_num,
                                            label_bytes)
                label_data = np.array(
                    label_array, dtype='int64').reshape(
                        (sample_info.label_frame_num, 1))

                feature_frame_num = sample_info.feature_frame_num
                feature_dim = sample_info.feature_dim
                assert feature_frame_num * feature_dim * 4 == len(feature_bytes)
                feature_array = struct.unpack('f' * feature_frame_num *
                                              feature_dim, feature_bytes)
                feature_data = np.array(
                    feature_array, dtype='float32').reshape((
                        sample_info.feature_frame_num, sample_info.feature_dim))

                sample_data = (feature_data, label_data)
                for transformer in self._transformers:
                    # @TODO(pkuyym) to make transfomer only accept feature_data
                    sample_data = transformer.perform_trans(sample_data)

                while order_id != out_order[0]:
                    time.sleep(0.001)

                # drop long sentence
                if self._drop_sentence_len >= sample_data[0].shape[0]:
                    sample_queue.put(sample_data)

                out_order[0] += 1
                ins = sample_info_queue.get()

            sample_queue.put(EpochEndSignal())

        out_order = self._manager.list([0])
        args = (sample_info_queue, sample_queue, out_order)
        workers = [
            Process(
                target=ordered_processing_worker, args=args)
            for _ in xrange(self._process_num)
        ]

        for w in workers:
            w.daemon = True
            w.start()

        finished_process_num = 0

        while finished_process_num < self._process_num:
            sample = sample_queue.get()
            if isinstance(sample, EpochEndSignal):
                finished_process_num += 1
                continue
            yield sample

        feeding_thread.join()
        for w in workers:
            w.join()

    def batch_iterator(self, batch_size, minimum_batch_size):
        batch_samples = []
        lod = [0]
        # check whether need parallel here
        for sample in self._sample_generator():
            batch_samples.append(sample)
            lod.append(lod[-1] + sample[0].shape[0])
            if len(batch_samples) == batch_size:
                batch_feature = np.zeros(
                    (lod[-1], self._frame_dim), dtype="float32")
                batch_label = np.zeros((lod[-1], 1), dtype="int64")
                start = 0
                for sample in batch_samples:
                    frame_num = sample[0].shape[0]
                    batch_feature[start:start + frame_num, :] = sample[0]
                    batch_label[start:start + frame_num, :] = sample[1]
                    start += frame_num
                yield (batch_feature, batch_label, lod)
                batch_samples = []
                lod = [0]

        if len(batch_samples) >= minimum_batch_size:
            batch_feature = np.zeros(
                (lod[-1], self._frame_dim), dtype="float32")
            batch_label = np.zeros((lod[-1], 1), dtype="int64")
            start = 0
            for sample in batch_samples:
                frame_num = sample[0].shape[0]
                batch_feature[start:start + frame_num, :] = sample[0]
                batch_label[start:start + frame_num, :] = sample[1]
                start += frame_num
            yield (batch_feature, batch_label, lod)
