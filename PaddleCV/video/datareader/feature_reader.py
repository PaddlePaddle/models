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
    This is for the three models: lstm, attention cluster, nextvlad

    dataset cfg: num_classes
                 batch_size
                 list
                 NextVlad only: eigen_file
    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.num_classes = cfg.MODEL.num_classes

        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']
        self.eigen_file = cfg.MODEL.get('eigen_file', None)
        self.seg_num = cfg.MODEL.get('seg_num', None)

    def create_reader(self):
        fl = open(self.filelist).readlines()
        fl = [line.strip() for line in fl if line.strip() != '']
        if self.mode == 'train':
            random.shuffle(fl)

        def reader():
            batch_out = []
            for filepath in fl:
                if python_ver < (3, 0):
                    data = pickle.load(open(filepath, 'rb'))
                else:
                    data = pickle.load(open(filepath, 'rb'), encoding='bytes')
                indexes = list(range(len(data)))
                if self.mode == 'train':
                    random.shuffle(indexes)
                for i in indexes:
                    record = data[i]
                    nframes = record[b'nframes']
                    rgb = record[b'feature'].astype(float)
                    audio = record[b'audio'].astype(float)
                    if self.mode != 'infer':
                        label = record[b'label']
                        one_hot_label = make_one_hot(label, self.num_classes)
                    video = record[b'video']

                    rgb = rgb[0:nframes, :]
                    audio = audio[0:nframes, :]

                    if self.name != 'NEXTVLAD':
                        rgb = dequantize(
                            rgb,
                            max_quantized_value=2.,
                            min_quantized_value=-2.)
                        audio = dequantize(
                            audio,
                            max_quantized_value=2,
                            min_quantized_value=-2)

                    if self.name == 'ATTENTIONCLUSTER':
                        sample_inds = generate_random_idx(rgb.shape[0],
                                                          self.seg_num)
                        rgb = rgb[sample_inds]
                        audio = audio[sample_inds]
                    if self.mode != 'infer':
                        batch_out.append((rgb, audio, one_hot_label))
                    else:
                        batch_out.append((rgb, audio, video))
                    if len(batch_out) == self.batch_size:
                        yield batch_out
                        batch_out = []

        return reader


def dequantize(feat_vector, max_quantized_value=2., min_quantized_value=-2.):
    """
    Dequantize the feature from the byte format to the float format
    """

    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value

    return feat_vector * scalar + bias


def make_one_hot(label, dim=3862):
    one_hot_label = np.zeros(dim)
    one_hot_label = one_hot_label.astype(float)
    for ind in label:
        one_hot_label[int(ind)] = 1
    return one_hot_label


def generate_random_idx(feature_len, seg_num):
    idxs = []
    stride = float(feature_len) / seg_num
    for i in range(seg_num):
        pos = (i + np.random.random()) * stride
        idxs.append(min(feature_len - 1, int(pos)))
    return idxs
