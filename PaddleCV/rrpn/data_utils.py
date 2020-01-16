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
from PIL import Image


class DatasetPath(object):
    def __init__(self, mode, dataset_name):
        self.mode = mode
        self.data_dir = dataset_name

    def get_data_dir(self):
        if self.mode == 'train':
            return os.path.join(self.data_dir, 'ch4_training_images')
        elif self.mode == 'val':
            return os.path.join(self.data_dir, 'ch4_test_images')

    def get_file_list(self):
        if self.mode == 'train':
            return os.path.join(self.data_dir,
                                'ch4_training_localization_transcription_gt')
        elif self.mode == 'val':
            return os.path.join(self.data_dir,
                                'ch4_test_localization_transcription_gt')


def get_image_blob(roidb, mode):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    if mode == 'train' or mode == 'val':
        with open(roidb['image'], 'rb') as f:
            data = f.read()
        data = np.frombuffer(data, dtype='uint8')
        img = cv2.imdecode(data, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_boxes = roidb['boxes']
        gt_label = roidb['gt_classes']
        # resize
        if mode == 'train':
            img, im_scale = _resize(img, target_size=800, max_size=1333)
            need_gt_boxes = gt_boxes.copy()
            need_gt_boxes[:, :4] *= im_scale
            img, need_gt_boxes, need_gt_label = _rotation(
                img, need_gt_boxes, gt_label, prob=1.0, gt_margin=1.4)
        else:
            img, im_scale = _resize(img, target_size=1000, max_size=1778)
            need_gt_boxes = gt_boxes
            need_gt_label = gt_label
        img = img.astype(np.float32, copy=False)
        img = img / 255.0
        mean = np.array(cfg.pixel_means)[np.newaxis, np.newaxis, :]
        std = np.array(cfg.pixel_std)[np.newaxis, np.newaxis, :]
        img -= mean
        img /= std
        img = img.transpose((2, 0, 1))
        return img, im_scale, need_gt_boxes, need_gt_label


def _get_size_scale(w, h, min_size, max_size=None):
    size = min_size
    scale = 1.0
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w), scale
    if w < h:
        ow = size
        oh = int(size * h / w)
        scale = size / w
    else:
        oh = size
        ow = int(size * w / h)
        scale = size / h
    scale = ow / w
    return (oh, ow), scale


def _resize(im, target_size=800, max_size=1333):
    if not isinstance(im, np.ndarray):
        raise TypeError("{}: image type is not numpy.")
    if len(im.shape) != 3:
        raise ImageError('{}: image is not 3-dimensional.')
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    selected_size = target_size
    if float(im_size_min) == 0:
        raise ZeroDivisionError('min size of image is 0')
    if max_size != 0:
        im_scale = float(selected_size) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = im_scale
        im_scale_y = im_scale

        resize_w = np.round(im_scale_x * float(im_shape[1]))
        resize_h = np.round(im_scale_y * float(im_shape[0]))
        im_info = [resize_h, resize_w, im_scale]
    else:
        im_scale_x = float(selected_size) / float(im_shape[1])
        im_scale_y = float(selected_size) / float(im_shape[0])

        resize_w = selected_size
        resize_h = selected_size

    im = Image.fromarray(im)
    im = im.resize((int(resize_w), int(resize_h)), 2)
    im = np.array(im)
    return im, im_scale_x


def _rotation(image,
              gt_boxes,
              gt_label,
              prob,
              fixed_angle=-1,
              r_range=(360, 0),
              gt_margin=1.4):
    rotate_range = r_range[0]
    shift = r_range[1]
    angle = np.array([np.max([0, fixed_angle])])
    if np.random.rand() <= prob:
        angle = np.array(
            np.random.rand(1) * rotate_range - shift, dtype=np.int16)
    '''
    rotate image
    '''
    image = np.array(image)
    (h, w) = image.shape[:2]
    scale = 1.0
    # set the rotation center
    center = (w / 2, h / 2)
    # anti-clockwise angle in the function
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, M, (w, h))
    # back to PIL image
    im_width, im_height = w, h
    '''
    rotate boxes
    '''
    need_gt_boxes = gt_boxes.copy()
    origin_gt_boxes = need_gt_boxes
    rotated_gt_boxes = np.empty((len(need_gt_boxes), 5), dtype=np.float32)
    # anti-clockwise to clockwise arc
    cos_cita = np.cos(np.pi / 180 * angle)
    sin_cita = np.sin(np.pi / 180 * angle)
    # clockwise matrix
    rotation_matrix = np.array([[cos_cita, sin_cita], [-sin_cita, cos_cita]])
    pts_ctr = origin_gt_boxes[:, 0:2]
    pts_ctr = pts_ctr - np.tile((im_width / 2, im_height / 2),
                                (gt_boxes.shape[0], 1))
    pts_ctr = np.array(np.dot(pts_ctr, rotation_matrix), dtype=np.int16)
    pts_ctr = np.squeeze(
        pts_ctr, axis=-1) + np.tile((im_width / 2, im_height / 2),
                                    (gt_boxes.shape[0], 1))
    origin_gt_boxes[:, 0:2] = pts_ctr
    len_of_gt = len(origin_gt_boxes)
    # rectificate the angle in the range of [-45, 45]
    for idx in range(len_of_gt):
        ori_angle = origin_gt_boxes[idx, 4]
        height = origin_gt_boxes[idx, 3]
        width = origin_gt_boxes[idx, 2]
        # step 1: normalize gt (-45,135)
        if width < height:
            ori_angle += 90
            width, height = height, width
        # step 2: rotate (-45,495)
        rotated_angle = ori_angle + angle
        # step 3: normalize rotated_angle (-45,135)
        while rotated_angle > 135:
            rotated_angle = rotated_angle - 180
        rotated_gt_boxes[idx, 0] = origin_gt_boxes[idx, 0]
        rotated_gt_boxes[idx, 1] = origin_gt_boxes[idx, 1]
        rotated_gt_boxes[idx, 3] = height * gt_margin
        rotated_gt_boxes[idx, 2] = width * gt_margin
        rotated_gt_boxes[idx, 4] = rotated_angle
    x_inbound = np.logical_and(rotated_gt_boxes[:, 0] >= 0,
                               rotated_gt_boxes[:, 0] < im_width)
    y_inbound = np.logical_and(rotated_gt_boxes[:, 1] >= 0,
                               rotated_gt_boxes[:, 1] < im_height)
    inbound = np.logical_and(x_inbound, y_inbound)
    need_gt_boxes = rotated_gt_boxes[inbound]
    need_gt_label = gt_label.copy()
    need_gt_label = need_gt_label[inbound]
    return image, need_gt_boxes, need_gt_label


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
