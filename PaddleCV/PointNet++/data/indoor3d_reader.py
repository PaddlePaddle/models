# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp
import numpy as np
import h5py
import random


class Indoor3DReader(object):
    def __init__(self, data_dir, test_area="Area_5"):
        self.data_dir = data_dir
        self.test_area = test_area
        self.load_data()

    def _read_data_file(self, fname):
        assert osp.isfile(fname), \
                "{} is not a file".format(fname)
        with open(fname) as f:
            return [line.strip() for line in f]

    def _load_h5_file(self, fname):
        assert osp.isfile(fname), \
                "{} is not a file".format(fname)
        f = h5py.File(fname)
        return f['data'][:], f['label'][:]

    def load_data(self):
        # read all_files.txt
        all_files_fname = osp.join(self.data_dir, 'all_files.txt')
        all_files = self._read_data_file(all_files_fname)

        # read room_filelist.txt
        room_fname = osp.join(self.data_dir, 'room_filelist.txt')
        room_filelist = self._read_data_file(room_fname)

        points, labels = [], []
        for f in all_files:
            h5_fname = osp.join(self.data_dir, osp.split(f)[-1]) 
            point, label = self._load_h5_file(h5_fname)
            points.append(point)
            labels.append(label)
        points = np.concatenate(points, 0)
        labels = np.concatenate(labels, 0)

        train_idxs, test_idxs = [], []
        for i, room in enumerate(room_filelist):
            if self.test_area in room:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        self.data = {}
        self.data['train'] = {}
        self.data['train']['points'] = points[train_idxs, ...]
        self.data['train']['labels'] = labels[train_idxs, ...]
        self.data['test'] = {}
        self.data['test']['points'] = points[test_idxs, ...]
        self.data['test']['labels'] = labels[test_idxs, ...]

    def get_reader(self, batch_size, num_points, mode='train', shuffle=True):
        assert mode in ['train', 'test'], \
                "mode can only be 'train' or 'test'"
        data = self.data[mode]
        points = data['points']
        labels = data['labels']

        if shuffle:
            idxs = np.arange(len(points))
            np.random.shuffle(idxs)
            points = points[idxs]
            labels = labels[idxs]

        def reader():
            batch_out = []
            for point, label in zip(points, labels):
                # shuffle points
                p = point.copy()
                l = label.copy()
                pt_idxs = np.arange(num_points)
                np.random.shuffle(pt_idxs)
                p = p[pt_idxs]
                l = l[pt_idxs]

                xyz = p[:, :3]
                feature = p[:, 3:]
                batch_out.append((xyz, feature, l))

                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []

        return reader

if __name__ == "__main__":
    ir = Indoor3DReader("dataset/Indoor3DSemSeg/indoor3d_sem_seg_hdf5_data")

    train_reader = ir.get_reader(32, 4096)
    for i, data in enumerate(train_reader()):
        if i == 0:
            print(data[0][0].shape, data[0][1].shape, data[0][2].shape)
        print('train', i, len(data))

    test_reader = ir.get_reader(32, 4096, 'test')
    for i, data in enumerate(test_reader()):
        print('test', i, len(data))
