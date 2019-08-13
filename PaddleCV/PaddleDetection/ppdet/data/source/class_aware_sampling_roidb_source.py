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

import copy
import collections
import pickle as pkl
from ..dataset import Dataset


class ClassAwareSamplingRoiDbSource(Dataset):
    """ interface to load class aware sampling roidb data from files
    """

    def __init__(self,
                 anno_file,
                 image_dir=None,
                 samples=-1,
                 is_shuffle=True,
                 load_img=False,
                 cname2cid=None,
                 use_default_label=None,
                 mixup_epoch=-1,
                 with_background=True):
        """ Init

        Args:
            fname (str): label file path
            image_dir (str): root dir for images
            samples (int): samples to load, -1 means all
            is_shuffle (bool): whether to shuffle samples
            load_img (bool): whether load data in this class
            cname2cid (dict): the label name to id dictionary
            use_default_label (bool):whether use the default mapping of label to id
            mixup_epoch (int): parse mixup in first n epoch
            with_background (bool): whether load background 
                                    as a class
            random_img(bool): in random way or sequential way
        """
        super(ClassAwareSamplingRoiDbSource, self).__init__()
        self._epoch = -1
        assert os.path.isfile(anno_file) or os.path.isdir(
            anno_file), 'invalid file[%s] for ClassAwareSamplingRoidbSource' % (
                anno_file)
        self._fname = anno_file
        self._image_dir = image_dir
        if image_dir is not None:
            assert os.path.isdir(image_dir), 'invalid image directory[%s]' % (
                image_dir)
        self._roidb = None
        self.cls_img_id_map = None
        self.img_id_sampler = None
        self.classes = None
        self._drained = False
        self._samples = samples
        self._load_img = load_img
        self.use_default_label = use_default_label
        self._with_background = with_background
        self.cname2cid = cname2cid
        self.random_img = True

    def __str__(self):
        return 'ClassAwareSamplingRoidbSource(fname:%s,epoch:%d,size:%d)' \
            % (self._fname, self._epoch, self.size())

    def next(self):
        """ load next sample
        """
        if self._epoch < 0:
            self.reset()

        sample_class = random.choice(self.classes)
        img_id = self.img_id_sampler[sample_class].next()

        _pos = self.img_id_pos_map[img_id]
        sample = copy.deepcopy(self._roidb[_pos])
        if self._load_img:
            sample['image'] = self._load_image(sample['im_file'])
        else:
            sample['im_file'] = os.path.join(self._image_dir, sample['im_file'])

        return sample

    def _load(self):
        """ load data from file
        """
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
        """ implementation of Dataset.reset
        """
        if self._roidb is None:
            self._roidb = self._load()

            img_id_pos_map = {}
            img_id_sampler = {}
            for i, roidb in enumerate(self._roidb):
                img_id_pos_map[roidb['im_id'][0]] = i
                for gt_cls in roidb['gt_class'][:, 0]:
                    img_id_sampler.setdefault(
                        gt_cls, ImgSampler(
                            random=self.random_img)).append(roidb['im_id'][0])
            self.img_id_pos_map = img_id_pos_map
            self.img_id_sampler = img_id_sampler
            self.classes = list(img_id_sampler.keys())

        self._samples = len(self._roidb)

        if self._epoch < 0:
            self._epoch = 0

    def size(self):
        """ implementation of Dataset.size
        """
        return len(self._roidb)

    def drained(self):
        """ implementation of Dataset.drained
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return False

    def epoch_id(self):
        """ return epoch id for latest sample
        """
        return self._epoch


class ImgSampler(object):
    def __init__(self, random=True):
        super(ImgSampler, self).__init__()
        self.img_ids = []
        self.pos = 0
        self.random = random

    def __len__(self):
        return len(self.img_ids)

    def append(self, img_id):
        self.img_ids.append(img_id)

    def next(self):
        if self.random:
            img_id = random.choice(self.img_ids)
        else:
            img_id = self.img_ids[self.pos]
            self.pos += 1
            if self.pos >= len(self):
                self.reset()
        return img_id

    def reset(self):
        random.shuffle(self.img_ids)
        self.pos = 0
