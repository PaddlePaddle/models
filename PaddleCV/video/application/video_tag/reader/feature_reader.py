#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import sys
from .reader_utils import DataReader
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import random

python_ver = sys.version_info


class FeatureReader(DataReader):
    """
    Data reader for youtube-8M dataset, which was stored as features extracted by prior networks
    This is for the three models: lstm

    dataset cfg: num_classes
                 batch_size
                 list
    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.num_classes = cfg.MODEL.num_classes

        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']
        self.seg_num = cfg.MODEL.get('seg_num', None)

    def create_reader(self):
        fl = open(self.filelist).readlines()
        fl = [line.strip() for line in fl if line.strip() != '']
        if self.mode == 'train':
            random.shuffle(fl)

        def reader():
            batch_out = []
            for item in fl:
                fileinfo = item.split(' ')
                filepath = fileinfo[0]
                rgb = np.load(filepath, allow_pickle=True)
                nframes = rgb.shape[0]
                label = [int(i) for i in fileinfo[1:]]
                one_hot_label = make_one_hot(label, self.num_classes)

                if self.mode != 'infer':
                    batch_out.append((rgb, one_hot_label))
                else:
                    batch_out.append((rgb, filepath.split('/')[-1]))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return reader


def make_one_hot(label, dim=3862):
    one_hot_label = np.zeros(dim)
    one_hot_label = one_hot_label.astype(float)
    for ind in label:
        one_hot_label[int(ind)] = 1
    return one_hot_label
