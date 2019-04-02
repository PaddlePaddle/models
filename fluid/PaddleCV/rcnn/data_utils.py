# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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
#
# Based on:
# --------------------------------------------------------
# Detectron
# Copyright (c) 2017-present, Facebook, Inc.
# Licensed under the Apache License, Version 2.0;
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
from config import cfg
import os


class DatasetPath(object):
    def __init__(self, mode):
        self.mode = mode
        mode_name = 'train' if mode == 'train' else 'val'
        if cfg.dataset != 'coco2014' and cfg.dataset != 'coco2017':
            raise NotImplementedError('Dataset {} not supported'.format(
                cfg.dataset))
        self.sub_name = mode_name + cfg.dataset[-4:]

    def get_data_dir(self):
        return os.path.join(cfg.data_dir, self.sub_name)

    def get_file_list(self):
        sfile_list = 'annotations/instances_' + self.sub_name + '.json'
        return os.path.join(cfg.data_dir, sfile_list)


def get_image_blob(roidb, mode):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    if mode == 'train':
        scales = cfg.TRAIN.scales
        scale_ind = np.random.randint(0, high=len(scales))
        target_size = scales[scale_ind]
        max_size = cfg.TRAIN.max_size
    else:
        target_size = cfg.TEST.scales[0]
        max_size = cfg.TEST.max_size
    im = cv2.imread(roidb['image'])
    try:
        assert im is not None
    except AssertionError as e:
        print('Failed to read image \'{}\''.format(roidb['image']))
        os._exit(0)
    if roidb['flipped']:
        im = im[:, ::-1, :]
    im, im_scale = prep_im_for_blob(im, cfg.pixel_means, target_size, max_size)

    return im, im_scale


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    im -= pixel_means

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    im_height, im_width, channel = im.shape
    channel_swap = (2, 0, 1)  #(batch, channel, height, width)
    im = im.transpose(channel_swap)
    return im, im_scale
