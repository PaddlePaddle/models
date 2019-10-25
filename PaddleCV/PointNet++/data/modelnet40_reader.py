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
import logging

__all__ = ["ModelNet40ClsReader"]

logger = logging.getLogger(__name__)


class ModelNet40ClsReader(object):
    def __init__(self, data_dir, mode='train', transforms=None):
        assert mode in ['train', 'test'], \
                "mode can only be 'train' or 'test'"
        self.data_dir = data_dir
        self.mode = mode 
        self.transforms = transforms
        self.load_data()

    def _read_data_file(self, fname):
        assert osp.isfile(fname), \
                "{} is not a file".format(fname)
        with open(fname) as f:
            return [line.strip()[5:] for line in f]

    def _load_h5_file(self, fname):
        assert osp.isfile(fname), \
                "{} is not a file".format(fname)
        f = h5py.File(fname, mode='r')
        return f['data'][:], f['label'][:]

    def load_data(self):
        logger.info("Loading ModelNet40 dataset {} split from {} "
                    "...".format(self.mode, self.data_dir))
        if self.mode == 'train':
            files_fname = osp.join(self.data_dir, 'train_files.txt')
            files = self._read_data_file(files_fname)
        else:
            files_fname = osp.join(self.data_dir, 'test_files.txt')
            files = self._read_data_file(files_fname)

        points, labels = [], []
        for f in files:
            h5_fname = osp.join(self.data_dir, osp.split(f)[-1])
            point, label = self._load_h5_file(h5_fname)
            points.append(point)
            labels.append(label)
        self.points = np.concatenate(points, 0)
        self.labels = np.concatenate(labels, 0)
        logger.info("Load {} data finished".format(self.mode))

    def get_reader(self, batch_size, num_points, shuffle=True):
        self.actual_number_of_points = min(max(np.random.randint(num_points * 0.8, num_points*1.2),1),
                                      self.points.shape[1])

        points = self.points
        labels = self.labels
        if shuffle and self.mode == 'train':
            idxs = np.arange(len(self.points))
            np.random.shuffle(idxs)
            points = points[idxs]
            labels = labels[idxs]

        def reader():
            batch_out = []
            for point, label in zip(points, labels):
                p = point.copy()
                l = label.copy()
                pt_idxs = np.arange(self.actual_number_of_points)
                if shuffle:
                    np.random.shuffle(pt_idxs)
                c_points = p[pt_idxs]
                if self.transforms is not None:
                    for trans in self.transforms:
                        c_points = trans(c_points)
                
                xyz = c_points[:, :3]
                feature = c_points[:, 3:]
                label = l[:, np.newaxis]
                # batch_out.append((xyz, feature, label))
                batch_out.append((xyz, label))

                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []
        return reader

if __name__ == "__main__":
    import data_utils as d_utils

    trans_list = [
        d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
        d_utils.PointcloudRotatePerturbation(),
    ]




    train_ir = ModelNet40ClsReader("modelnet40_ply_hdf5_2048",train=True,transforms=trans_list)
    test_ir = ModelNet40ClsReader("modelnet40_ply_hdf5_2048",train=False)

    train_reader = train_ir.get_reader(1, 16)

    # test
    a = next(train_reader())
    print("train",a,len(a[0][0]))

    """
    for i, data in enumerate(train_reader()):
        print('train', i, len(data))

    test_reader = test_ir.get_reader(32, 4096)
    for i, data in enumerate(test_reader()):
        print('test', i, len(data))
    """
