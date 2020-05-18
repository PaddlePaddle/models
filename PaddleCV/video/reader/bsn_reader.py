#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import platform
import random
import numpy as np
import pandas as pd
import multiprocessing
import json
import logging
import functools
import paddle
import paddle.fluid as fluid
logger = logging.getLogger(__name__)

from .reader_utils import DataReader
from models.bsn.bsn_utils import iou_with_anchors, ioa_with_anchors


class BSNVideoReader(DataReader):
    """
    Data reader for BsnTem model, which was stored as features extracted by prior networks
    dataset cfg: anno_file, annotation file path,
                 feat_path, feature path,
                 file_list, file list for infer,
                 tscale, temporal length of input,
                 anchor_xmin, anchor_xmax, the range of each point in the feature sequence,
                 batch_size, batch size of input data,
                 num_threads, number of threads of data processing

    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.tscale = cfg.MODEL.tscale  # 100
        self.gt_boundary_ratio = cfg.MODEL.gt_boundary_ratio
        self.anno_file = cfg.MODEL.anno_file
        self.file_list = cfg.INFER.filelist
        self.subset = cfg[mode.upper()]['subset']
        self.tgap = 1. / self.tscale
        self.feat_path = cfg.MODEL.feat_path
        self.anchor_xmin = [self.tgap * i for i in range(self.tscale)]
        self.anchor_xmax = [self.tgap * i for i in range(1, self.tscale + 1)]
        self.get_dataset_dict()

        self.batch_size = cfg[mode.upper()]['batch_size']
        self.num_threads = cfg[mode.upper()]['num_threads']
        if (mode == 'test') or (mode == 'infer'):
            self.num_threads = 1  # set num_threads as 1 for test and infer

    def get_dataset_dict(self):
        self.video_dict = {}
        if self.mode == "infer":
            annos = json.load(open(self.file_list))
            for video_name in annos.keys():
                self.video_dict[video_name] = annos[video_name]
        else:
            annos = json.load(open(self.anno_file))
            for video_name in annos.keys():
                video_subset = annos[video_name]["subset"]
                if self.subset == "train_val":
                    if "train" in video_subset or "validation" in video_subset:
                        self.video_dict[video_name] = annos[video_name]
                else:
                    if self.subset in video_subset:
                        self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()
        print("%s subset video numbers: %d" %
              (self.subset, len(self.video_list)))

    def get_video_label(self, video_name):
        video_info = self.video_dict[video_name]
        video_second = video_info['duration_second']
        video_labels = video_info['annotations']

        gt_bbox = []
        for gt in video_labels:
            tmp_start = max(min(1, gt["segment"][0] / video_second), 0)
            tmp_end = max(min(1, gt["segment"][1] / video_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = np.maximum(self.tgap, self.gt_boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack(
            (gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack(
            (gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

        match_score_action = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_action.append(
                np.max(
                    ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[
                        jdx], gt_xmins, gt_xmaxs)))
        match_score_start = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_start.append(
                np.max(
                    ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[
                        jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(self.anchor_xmin)):
            match_score_end.append(
                np.max(
                    ioa_with_anchors(self.anchor_xmin[jdx], self.anchor_xmax[
                        jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))

        gt_start = np.array(match_score_start)
        gt_end = np.array(match_score_end)
        gt_action = np.array(match_score_action)
        return gt_start, gt_end, gt_action

    def load_file(self, video_name):
        video_feat = np.load(self.feat_path + "/" + video_name + ".npy")
        video_feat = video_feat.T
        video_feat = video_feat.astype("float32")
        return video_feat

    def create_reader(self):
        """reader creator for ctcn model"""
        if self.mode == 'infer':
            return self.make_infer_reader()
        if self.num_threads == 1:
            return self.make_reader()
        else:
            sysstr = platform.system()
            if sysstr == 'Windows':
                return self.make_multithread_reader()
            else:
                return self.make_multiprocess_reader()

    def make_infer_reader(self):
        """reader for inference"""

        def reader():
            batch_out = []
            for video_name in self.video_list:
                video_idx = self.video_list.index(video_name)
                video_feat = self.load_file(video_name)
                batch_out.append((video_feat, video_idx))

                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return reader

    def make_reader(self):
        """single process reader"""

        def reader():
            video_list = self.video_list
            if self.mode == 'train':
                random.shuffle(video_list)

            batch_out = []
            for video_name in video_list:
                video_idx = video_list.index(video_name)
                video_feat = self.load_file(video_name)
                gt_start, gt_end, gt_action = self.get_video_label(video_name)

                if self.mode == 'train' or self.mode == 'valid':
                    batch_out.append((video_feat, gt_start, gt_end, gt_action))
                elif self.mode == 'test':
                    batch_out.append(
                        (video_feat, gt_start, gt_end, gt_action, video_idx))
                else:
                    raise NotImplementedError('mode {} not implemented'.format(
                        self.mode))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return reader

    def make_multithread_reader(self):
        def reader():
            if self.mode == 'train':
                random.shuffle(self.video_list)

            for video_name in self.video_list:
                video_idx = self.video_list.index(video_name)
                yield [video_name, video_idx]

        def process_data(sample, mode):
            video_name = sample[0]
            video_idx = sample[1]
            video_feat = self.load_file(video_name)
            gt_start, gt_end, gt_action = self.get_video_label(video_name)
            if mode == 'train' or mode == 'valid':
                return (video_feat, gt_start, gt_end, gt_action)
            elif mode == 'test':
                return (video_feat, gt_start, gt_end, gt_action, video_idx)
            else:
                raise NotImplementedError('mode {} not implemented'.format(
                    self.mode))

        mapper = functools.partial(process_data, mode=self.mode)

        def batch_reader():
            xreader = fluid.io.xmap_readers(mapper, reader, self.num_threads,
                                            1024)
            batch = []
            for item in xreader():
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        return batch_reader

    def make_multiprocess_reader(self):
        """multiprocess reader"""

        def read_into_queue(video_list, queue):

            batch_out = []
            for video_name in video_list:
                video_idx = video_list.index(video_name)
                video_feat = self.load_file(video_name)
                gt_start, gt_end, gt_action = self.get_video_label(video_name)

                if self.mode == 'train' or self.mode == 'valid':
                    batch_out.append((video_feat, gt_start, gt_end, gt_action))
                elif self.mode == 'test':
                    batch_out.append(
                        (video_feat, gt_start, gt_end, gt_action, video_idx))
                else:
                    raise NotImplementedError('mode {} not implemented'.format(
                        self.mode))

                if len(batch_out) == self.batch_size:
                    queue.put(batch_out)
                    batch_out = []
            queue.put(None)

        def queue_reader():
            video_list = self.video_list
            if self.mode == 'train':
                random.shuffle(video_list)

            n = self.num_threads
            queue_size = 20
            reader_lists = [None] * n
            file_num = int(len(video_list) // n)
            for i in range(n):
                if i < len(reader_lists) - 1:
                    tmp_list = video_list[i * file_num:(i + 1) * file_num]
                else:
                    tmp_list = video_list[i * file_num:]
                reader_lists[i] = tmp_list

            queue = multiprocessing.Queue(queue_size)
            p_list = [None] * len(reader_lists)
            # for reader_list in reader_lists:
            for i in range(len(reader_lists)):
                reader_list = reader_lists[i]
                p_list[i] = multiprocessing.Process(
                    target=read_into_queue, args=(reader_list, queue))
                p_list[i].start()
            reader_num = len(reader_lists)
            finish_num = 0
            while finish_num < reader_num:
                sample = queue.get()
                if sample is None:
                    finish_num += 1
                else:
                    yield sample
            for i in range(len(p_list)):
                if p_list[i].is_alive():
                    p_list[i].join()

        return queue_reader


class BSNProposalReader(DataReader):
    """
    Data reader for BsnPem model, which was stored as features extracted by prior networks
    dataset cfg: anno_file, annotation file path,
                 file_list, file list for infer,
                 top_K, number of proposals during training/test,
                 feat_path, feature path generated by PGM,
                 prop_path, proposal path generated by PGM,
                 batch_size, batch size of input data,
                 num_threads, number of threads of data processing.
    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode

        self.top_K = cfg[mode.upper()]['top_K']
        self.anno_file = cfg.MODEL.anno_file
        self.file_list = cfg.INFER.filelist
        self.subset = cfg[mode.upper()]['subset']

        if mode == 'infer':
            self.feat_path = cfg[mode.upper()]['feat_path']
            self.prop_path = cfg[mode.upper()]['prop_path']
        else:
            self.feat_path = cfg.MODEL.feat_path
            self.prop_path = cfg.MODEL.prop_path
        self.get_dataset_dict()

        self.batch_size = cfg[mode.upper()]['batch_size']
        self.num_threads = cfg[mode.upper()]['num_threads']
        if (mode == 'test') or (mode == 'infer'):
            self.num_threads = 1  # set num_threads as 1 for test and infer

    def get_dataset_dict(self):
        self.video_dict = {}
        if self.mode == "infer":
            annos = json.load(open(self.file_list))
            for video_name in annos.keys():
                self.video_dict[video_name] = annos[video_name]
        else:
            annos = json.load(open(self.anno_file))
            for video_name in annos.keys():
                video_subset = annos[video_name]["subset"]
                if self.subset in video_subset:
                    self.video_dict[video_name] = annos[video_name]
        self.video_list = list(self.video_dict.keys())
        self.video_list.sort()
        print("%s subset video numbers: %d" %
              (self.subset, len(self.video_list)))

    def get_props(self, video_name):
        pdf = pd.read_csv(self.prop_path + video_name + ".csv")
        pdf = pdf[:self.top_K]
        props_start = pdf.xmin.values[:]
        props_end = pdf.xmax.values[:]
        props_start_score = pdf.xmin_score.values[:]
        props_end_score = pdf.xmax_score.values[:]
        props_info = np.hstack([
            props_start[:, np.newaxis], props_end[:, np.newaxis],
            props_start_score[:, np.newaxis], props_end_score[:, np.newaxis]
        ])
        if self.mode == "infer":
            return props_info
        else:
            props_iou = pdf.match_iou.values[:]
            return props_iou, props_info

    def load_file(self, video_name):
        video_feat = np.load(self.feat_path + video_name + ".npy")
        video_feat = video_feat[:self.top_K, :]
        video_feat = video_feat.astype("float32")
        return video_feat

    def create_reader(self):
        """reader creator for ctcn model"""
        if self.mode == 'infer':
            return self.make_infer_reader()
        if self.num_threads == 1:
            return self.make_reader()
        else:
            sysstr = platform.system()
            if sysstr == 'Windows':
                return self.make_multithread_reader()
            else:
                return self.make_multiprocess_reader()

    def make_infer_reader(self):
        """reader for inference"""

        def reader():
            batch_out = []
            for video_name in self.video_list:
                video_idx = self.video_list.index(video_name)
                props_feat = self.load_file(video_name)
                props_info = self.get_props(video_name)
                batch_out.append((props_feat, props_info, video_idx))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return reader

    def make_reader(self):
        """single process reader"""

        def reader():
            video_list = self.video_list
            if self.mode == 'train':
                random.shuffle(video_list)

            batch_out = []
            for video_name in video_list:
                video_idx = video_list.index(video_name)
                props_feat = self.load_file(video_name)
                props_iou, props_info = self.get_props(video_name)

                if self.mode == 'train' or self.mode == 'valid':
                    batch_out.append((props_feat, props_iou))
                elif self.mode == 'test':
                    batch_out.append(
                        (props_feat, props_iou, props_info, video_idx))
                else:
                    raise NotImplementedError('mode {} not implemented'.format(
                        self.mode))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return reader

    def make_multithread_reader(self):
        def reader():
            if self.mode == 'train':
                random.shuffle(self.video_list)
            for video_name in self.video_list:
                video_idx = self.video_list.index(video_name)
                yield [video_name, video_idx]

        def process_data(sample, mode):
            video_name = sample[0]
            video_idx = sample[1]
            props_feat = self.load_file(video_name)
            props_iou, props_info = self.get_props(video_name)
            if mode == 'train' or mode == 'valid':
                return (props_feat, props_iou)
            elif mode == 'test':
                return (props_feat, props_iou, props_info, video_idx)
            else:
                raise NotImplementedError('mode {} not implemented'.format(
                    mode))

        mapper = functools.partial(process_data, mode=self.mode)

        def batch_reader():
            xreader = fluid.io.xmap_readers(mapper, reader, self.num_threads,
                                            1024)
            batch = []
            for item in xreader():
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        return batch_reader

    def make_multiprocess_reader(self):
        """multiprocess reader"""

        def read_into_queue(video_list, queue):

            batch_out = []
            for video_name in video_list:
                video_idx = video_list.index(video_name)
                props_feat = self.load_file(video_name)
                props_iou, props_info = self.get_props(video_name)

                if self.mode == 'train' or self.mode == 'valid':
                    batch_out.append((props_feat, props_iou))
                elif self.mode == 'test':
                    batch_out.append(
                        (props_feat, props_iou, props_info, video_idx))
                else:
                    raise NotImplementedError('mode {} not implemented'.format(
                        self.mode))

                if len(batch_out) == self.batch_size:
                    queue.put(batch_out)
                    batch_out = []
            queue.put(None)

        def queue_reader():
            video_list = self.video_list
            if self.mode == 'train':
                random.shuffle(video_list)

            n = self.num_threads
            queue_size = 20
            reader_lists = [None] * n
            file_num = int(len(video_list) // n)
            for i in range(n):
                if i < len(reader_lists) - 1:
                    tmp_list = video_list[i * file_num:(i + 1) * file_num]
                else:
                    tmp_list = video_list[i * file_num:]
                reader_lists[i] = tmp_list

            queue = multiprocessing.Queue(queue_size)
            p_list = [None] * len(reader_lists)
            # for reader_list in reader_lists:
            for i in range(len(reader_lists)):
                reader_list = reader_lists[i]
                p_list[i] = multiprocessing.Process(
                    target=read_into_queue, args=(reader_list, queue))
                p_list[i].start()
            reader_num = len(reader_lists)
            finish_num = 0
            while finish_num < reader_num:
                sample = queue.get()
                if sample is None:
                    finish_num += 1
                else:
                    yield sample
            for i in range(len(p_list)):
                if p_list[i].is_alive():
                    p_list[i].join()

        return queue_reader
