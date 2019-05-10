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
#    interface to load data from local files and parse it for samples, 
#    eg: roidb data in pickled files

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import pickle as pkl
from ..dataset import Dataset

class RoiDbSource(Dataset):
    """ interface to load roidb data from files
    """
    def __init__(self, fnames,
        image_dir=None, samples=-1):
        super(RoiDbSource, self).__init__()
        self._epoch = -1
        self._fnames = [fnames] if type(fnames) is str else fnames
        for f in self._fnames:
            assert os.path.isfile(f) or os.path.isdir(f), 'invalid file[%s] for RoiDbSource' % (f)
        self._image_dir = image_dir
        if image_dir is not None:
            assert os.path.isdir(image_dir), 'invalid image directory[%s]' % (image_dir)

        self._roidb = None
        self._pos = -1
        self._drained = False
        self._samples = samples

    def __str__(self):
        return 'RoiDbSource(fname:%s,epoch:%d,size:%d,pos:%d)' \
            % (str(self._fnames), self._epoch, self.size(), self._pos)

    def next(self):
        """ load next sample
        """
        if self._epoch < 0:
            self.reset()

        if self._pos >= self.size():
            self._drained = True
            raise StopIteration('%s no more data' % (str(self)))
        else:
            sample = self._roidb[self._pos]
            if self._image_dir is not None:
                sample['image'] = self._load_image(sample['image_url'])
            self._pos += 1
            return sample

    def _load(self, fnames):
        """ load data from file
        """
        from .roidb_loader import load
        return load(fnames, self._samples)[0]

    def _load_image(self, where):
        fn = os.path.join(self._image_dir, where)
        with open(fn, 'rb') as f:
            return f.read()

    def reset(self):
        """ implementation of Dataset.reset
        """
        if self._roidb is None:
            self._roidb = self._load(self._fnames)
        random.shuffle(self._roidb)

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0
        self._drained = False

    def size(self):
        """ implementation of Dataset.size
        """
        return len(self._roidb)

    def drained(self):
        """ implementation of Dataset.drained
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._pos >= self.size()

