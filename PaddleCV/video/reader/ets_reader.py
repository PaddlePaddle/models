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
import random
import sys
import numpy as np
import functools
import paddle
import paddle.fluid as fluid

import logging
logger = logging.getLogger(__name__)

import pickle

from .reader_utils import DataReader

python_ver = sys.version_info


class ETSReader(DataReader):
    """
    Data reader for ETS model, which was stored as features extracted by prior networks
    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode

        self.feat_path = cfg.MODEL.feat_path
        self.dict_file = cfg.MODEL.dict_file
        self.START = cfg.MODEL.START
        self.END = cfg.MODEL.END
        self.UNK = cfg.MODEL.UNK

        self.filelist = cfg[mode.upper()]['filelist']
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.num_threads = cfg[mode.upper()]['num_threads']
        self.buffer_size = cfg[mode.upper()]['buffer_size']
        if (mode == 'test') or (mode == 'infer'):
            self.num_threads = 1  # set num_threads as 1 for test and infer

    def load_file(self):
        word_dict = dict()
        with open(self.dict_file, 'r') as f:
            for i, line in enumerate(f):
                word_dict[line.strip().split()[0]] = i
        return word_dict

    def create_reader(self):
        """reader creator for ets model"""
        if self.mode == 'infer':
            return self.make_infer_reader()
        else:
            return self.make_multiprocess_reader()

    def make_infer_reader(self):
        """reader for inference"""

        def reader():
            batch_out = []

            with open(self.filelist) as f:
                lines = f.readlines()
                reader_list = [
                    line.strip() for line in lines if line.strip() != ''
                ]

            word_dict = self.load_file()
            for line in reader_list:
                vid, stime, etime, sentence = line.split('\t')
                stime, etime = float(stime), float(etime)

                if python_ver < (3, 0):
                    datas = pickle.load(
                        open(os.path.join(self.feat_path, vid), 'rb'))
                else:
                    datas = pickle.load(
                        open(os.path.join(self.feat_path, vid), 'rb'),
                        encoding='bytes')

                feat = datas[int(stime * 5):int(etime * 5 + 0.5), :]
                init_ids = np.array([[0]], dtype='int64')
                init_scores = np.array([[0.]], dtype='float32')
                if feat.shape[0] == 0:
                    continue

                batch_out.append(
                    (feat, init_ids, init_scores, vid, stime, etime))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return reader

    def make_multiprocess_reader(self):
        """multiprocess reader"""

        def process_data(sample):
            vid, feat, stime, etime, sentence = sample

            if self.mode == 'train' or self.mode == 'valid':
                word_ids = [
                    word_dict.get(w, word_dict[self.UNK])
                    for w in sentence.split()
                ]
                word_ids_next = word_ids + [word_dict[self.END]]
                word_ids = [word_dict[self.START]] + word_ids
                return feat, word_ids, word_ids_next
            elif self.mode == 'test':
                init_ids = np.array([[0]], dtype='int64')
                init_scores = np.array([[0.]], dtype='float32')
                return feat, init_ids, init_scores, vid, stime, etime
            else:
                raise NotImplementedError('mode {} not implemented'.format(
                    self.mode))

        def make_reader():
            def reader():
                lines = open(self.filelist).readlines()
                reader_list = [
                    line.strip() for line in lines if line.strip() != ''
                ]
                if self.mode == 'train':
                    random.shuffle(reader_list)
                for line in reader_list:
                    vid, stime, etime, sentence = line.split('\t')
                    stime, etime = float(stime), float(etime)

                    if python_ver < (3, 0):
                        datas = pickle.load(
                            open(os.path.join(self.feat_path, vid), 'rb'))
                    else:
                        datas = pickle.load(
                            open(os.path.join(self.feat_path, vid), 'rb'),
                            encoding='bytes')

                    feat = datas[int(stime * 5):int(etime * 5 + 0.5), :]
                    if feat.shape[0] == 0:
                        continue

                    yield [vid, feat, stime, etime, sentence]

            mapper = functools.partial(process_data)

            return fluid.io.xmap_readers(mapper, reader, self.num_threads,
                                         self.buffer_size)

        def batch_reader():
            batch_out = []
            for out in _reader():
                batch_out.append(out)
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        word_dict = self.load_file()
        _reader = make_reader()
        return batch_reader
