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
#    interface to load data from txt file.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import copy
from ..dataset import Dataset


class SimpleSource(Dataset):
    """
    Load image files for testing purpose

    Args:
        test_file (str): list of image file names, relative to `image_dir`
        image_dir (str): root dir for images
        samples (int): number of samples to load, -1 means all
        load_img (bool): should images be loaded
    """

    def __init__(self,
                 test_file='',
                 image_dir=None,
                 samples=-1,
                 load_img=True,
                 **kwargs):
        super(SimpleSource, self).__init__()
        self._epoch = -1
        assert test_file != '' and os.path.isfile(test_file), \
            "test file not found: " + test_file
        self._fname = test_file
        self._image_dir = image_dir
        assert image_dir is not None and os.path.isdir(image_dir), \
            "image directory not found: " + image_dir
        self._simple = None
        self._pos = -1
        self._drained = False
        self._samples = samples
        self._load_img = load_img
        self._imid2path = {}

    def next(self):
        if self._epoch < 0:
            self.reset()

        if self._pos >= self.size():
            self._drained = True
            raise StopIteration("no more data in " + str(self))
        else:
            sample = copy.deepcopy(self._simple[self._pos])
            if self._load_img:
                sample['image'] = self._load_image(sample['im_file'])
            else:
                sample['im_file'] = os.path.join(self._image_dir,
                                                 sample['im_file'])
            self._pos += 1
            return sample

    def _load(self):
        assert os.path.isfile(self._fname) and self._fname.endswith('.txt'), \
            "invalid test file path"
        ct = 0
        records = []
        with open(self._fname, 'r') as fr:
            while True:
                line = fr.readline().strip()
                if not line or (self._samples > 0 and ct >= self._samples):
                    break
                rec = {'im_id': np.array([ct]), 'im_file': line}
                self._imid2path[ct] = line
                ct += 1
                records.append(rec)
        assert len(records) > 0, "no image file found in " + self._fname
        return records

    def _load_image(self, where):
        fn = os.path.join(self._image_dir, where)
        with open(fn, 'rb') as f:
            return f.read()

    def reset(self):
        if self._simple is None:
            self._simple = self._load()

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0
        self._drained = False

    def size(self):
        return len(self._simple)

    def drained(self):
        assert self._epoch >= 0, "the first epoch has not started yet"
        return self._pos >= self.size()

    def epoch_id(self):
        return self._epoch

    def get_imid2path(self):
        """return image id to image path map"""
        return self._imid2path
