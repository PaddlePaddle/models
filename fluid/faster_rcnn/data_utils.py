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


def get_image_blob(roidb, settings):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    scale_ind = np.random.randint(0, high=len(settings.scales))
    im = cv2.imread(roidb['image'])
    assert im is not None, \
        'Failed to read image \'{}\''.format(roidb['image'])
    if roidb['flipped']:
        im = im[:, ::-1, :]
    #print(im[10:, 10:, :])
    target_size = settings.scales[scale_ind]
    im, im_scale = prep_im_for_blob(im, settings.mean_value, target_size,
                                    settings.max_size)

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
    #print(im[10:, 10:, :])

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
    padding_im = np.zeros((max_size, max_size, im_shape[2]), dtype=np.float32)
    padding_im[:im_height, :im_width, :] = im
    #print(padding_im[10:, 10:, :])
    channel_swap = (2, 0, 1)  #(batch, channel, height, width)
    #im = im.transpose(channel_swap)
    padding_im = padding_im.transpose(channel_swap)
    #print(padding_im[10:, 10:, :])
    return padding_im, im_scale
