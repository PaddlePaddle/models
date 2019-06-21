# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# function:
#    interface to load data from local files and parse it for samples,
#    eg: roidb data in pickled files

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import copy

from ..dataset import Dataset


class RoiDbSource(Dataset):
    """
    Load roidb data from files

    Args:
        fname (str): label file path
        image_dir (str): root dir for images
        samples (int): samples to load, -1 means all
        is_shuffle (bool): whether to shuffle samples
        load_img (bool): should images be loaded
        cname2cid (dict): the label name to id dictionary
        use_default_label (bool):whether use the default mapping of label to id
        mixup_epoch (int): parse mixup in first n epoch
        with_background (bool): treat background as a separate class
    """

    def __init__(self,
                 anno_file='',
                 image_dir=None,
                 samples=-1,
                 is_shuffle=True,
                 load_img=False,
                 cname2cid=None,
                 use_default_label=True,
                 mixup_epoch=-1,
                 with_background=True):
        super(RoiDbSource, self).__init__()
        self._epoch = -1
        assert anno_file != '' and os.path.isfile(anno_file), \
            "annotation file not found: " + anno_file
        self._fname = anno_file
        self._image_dir = image_dir
        assert image_dir is not None and os.path.isdir(image_dir), \
            "image directory not found: " + image_dir
        self._roidb = None
        self._pos = -1
        self._drained = False
        self._samples = samples
        self._is_shuffle = is_shuffle
        self._load_img = load_img
        self.use_default_label = use_default_label
        self._mixup_epoch = mixup_epoch
        self._with_background = with_background
        self.cname2cid = cname2cid

    def next(self):
        if self._epoch < 0:
            self.reset()
        if self._pos >= self._samples:
            self._drained = True
            raise StopIteration("no more data in " + str(self))
        sample = copy.deepcopy(self._roidb[self._pos])
        if self._load_img:
            sample['image'] = self._load_image(sample['im_file'])
        else:
            sample['im_file'] = os.path.join(self._image_dir, sample['im_file'])

        if self._epoch < self._mixup_epoch:
            mix_idx = random.randint(1, self._samples - 1)
            mix_pos = (mix_idx + self._pos) % self._samples
            sample['mixup'] = copy.deepcopy(self._roidb[mix_pos])
            if self._load_image:
                sample['mixup']['image'] = \
                    self._load_image(sample['mixup']['im_file'])
            else:
                sample['mixup']['im_file'] = \
                    os.path.join(self._image_dir, sample['mixup']['im_file'])
        self._pos += 1
        return sample

    def _load(self):
        from . import loader
        records, cname2cid = loader.load(self._fname, self._samples,
                                         self._with_background, True,
                                         self.use_default_label, self.cname2cid)
        self.cname2cid = cname2cid
        return records

    def _load_image(self, where):
        fn = os.path.join(self._image_dir, where)
        with open(fn, 'rb') as f:
            return f.read()

    def reset(self):
        if self._roidb is None:
            self._roidb = self._load()

        self._samples = len(self._roidb)
        if self._is_shuffle:
            random.shuffle(self._roidb)

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0
        self._drained = False

    def size(self):
        return len(self._roidb)

    def drained(self):
        assert self._epoch >= 0, "the first epoch has not started yet"
        return self._pos >= self.size()

    def epoch_id(self):
        return self._epoch
