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

#function:
#    interface to load data from txt file.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import numpy as np
import copy
from ..dataset import Dataset


class SimpleSource(Dataset):
    """ a simple Dataset interface used to provide testing data which are image files stored in local files.
    """

    def __init__(self, test_file, image_dir=None, samples=-1, load_img=True):
        """ Init

        Args:
            test_file (str): file names for each image which is located in 'image_dir'
            image_dir (str): root dir for images
            samples (int): samples to load, -1 means all
            load_img (bool): whether load data in this class
        """
        super(SimpleSource, self).__init__()
        self._epoch = -1
        assert os.path.isfile(
            test_file), 'invalid file[%s] for SimpleSource' % (test_file)
        self._fname = test_file
        self._image_dir = image_dir
        if image_dir is not None:
            assert os.path.isdir(image_dir), 'invalid image directory[%s]' % (
                image_dir)
        self._simple = None
        self._pos = -1
        self._drained = False
        self._samples = samples
        self._load_img = load_img
        self._image_list = []

    def __str__(self):
        return 'SimpleSource(fname:%s,epoch:%d,size:%d,pos:%d)' \
            % (self._fname, self._epoch, self.size(), self._pos)

    def next(self):
        """ load next sample
        """
        if self._epoch < 0:
            self.reset()

        if self._pos >= self.size():
            self._drained = True
            raise StopIteration('%s no more data' % (str(self)))
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
        """ load data from file
        """
        assert os.path.isfile(self._fname) and self._fname.endswith(
            '.txt'), 'invalid test file path'
        ct = 0
        records = []
        with open(self._fname, 'r') as fr:
            while True:
                line = fr.readline().strip()
                if not line or (self._samples > 0 \
                        and ct >= self._samples):
                    break
                rec = {'im_id': np.array([ct]), 'im_file': line}
                self._image_list.append(line)
                ct += 1
                records.append(rec)
        assert len(records) > 0, 'not found any test image in %s' % (
            self._fname)
        return records

    def _load_image(self, where):
        fn = os.path.join(self._image_dir, where)
        with open(fn, 'rb') as f:
            return f.read()

    def reset(self):
        """ implementation of Dataset.reset
        """
        if self._simple is None:
            self._simple = self._load()

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0
        self._drained = False

    def size(self):
        """ implementation of Dataset.size
        """
        return len(self._simple)

    def drained(self):
        """ implementation of Dataset.drained
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._pos >= self.size()

    def epoch_id(self):
        """ return epoch id for latest sample
        """
        return self._epoch

    def get_image_list(self):
        """ return image file path list
        """
        return self._image_list
