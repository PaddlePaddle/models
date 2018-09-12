# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.utils.image_util import *
import random
from PIL import Image
from PIL import ImageDraw
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy
import six

from roidbs import JsonDataset
import data_utils


class Settings(object):
    def __init__(self, args=None):
        for arg, value in sorted(six.iteritems(vars(args))):
            setattr(self, arg, value)

        if 'coco2014' in args.dataset:
            self.class_nums = 81
            self.train_file_list = 'annotations/instances_train2014.json'
            self.train_data_dir = 'train2014'
            self.val_file_list = 'annotations/instances_val2014.json'
            self.val_data_dir = 'val2014'
        elif 'coco2017' in args.dataset:
            self.class_nums = 81
            self.train_file_list = 'annotations/instances_val2017.json'
            self.train_data_dir = 'val2017'
            self.val_file_list = 'annotations/instances_val2017.json'
            self.val_data_dir = 'val2017'
        else:
            raise NotImplementedError('Dataset {} not supported'.format(
                self.dataset))
        self.mean_value = np.array(self.mean_value)[
            np.newaxis, np.newaxis, :].astype('float32')


def coco(settings, mode, shuffle):
    if mode == 'train':
        settings.train_file_list = os.path.join(settings.data_dir,
                                                settings.train_file_list)
        settings.train_data_dir = os.path.join(settings.data_dir,
                                               settings.train_data_dir)
    elif mode == 'test':
        settings.val_file_list = os.path.join(settings.data_dir,
                                              settings.val_file_list)
        settings.val_data_dir = os.path.join(settings.data_dir,
                                             settings.val_data_dir)
    json_dataset = JsonDataset(settings, train=(mode == 'train'))
    roidbs = json_dataset.get_roidb()

    print("{} on {} with {} roidbs".format(mode, settings.dataset, len(roidbs)))

    def reader():
        if mode == "train" and shuffle:
            random.shuffle(roidbs)
        im_out, gt_boxes_out, gt_classes_out, is_crowd_out, im_info_out = [],[],[],[],[]
        lod = [0]
        for roidb in roidbs:
            im, im_scales = data_utils.get_image_blob(roidb, settings)
            im_height = np.round(roidb['height'] * im_scales)
            im_width = np.round(roidb['width'] * im_scales)
            im_info = np.array(
                [im_height, im_width, im_scales], dtype=np.float32)
            gt_boxes = roidb['gt_boxes'].astype('float32')
            gt_classes = roidb['gt_classes'].astype('int32')
            is_crowd = roidb['is_crowd'].astype('int32')
            if gt_boxes.shape[0] == 0:
                continue
            im_out.append(im)
            gt_boxes_out.extend(gt_boxes)
            gt_classes_out.extend(gt_classes)
            is_crowd_out.extend(is_crowd)
            im_info_out.append(im_info)
            lod.append(lod[-1] + gt_boxes.shape[0])
            if len(im_out) == settings.batch_size:
                im_out = np.array(im_out).astype('float32')
                gt_boxes_out = np.array(gt_boxes_out).astype('float32')
                gt_classes_out = np.array(gt_classes_out).astype('int32')
                is_crowd_out = np.array(is_crowd_out).astype('int32')
                im_info_out = np.array(im_info_out).astype('float32')
                yield im_out, gt_boxes_out, gt_classes_out, is_crowd_out, im_info_out, lod
                im_out, gt_boxes_out, gt_classes_out, is_crowd_out, im_info_out = [],[],[],[],[]
                lod = [0]

    return reader


def train(settings, shuffle=True):
    return coco(settings, 'train', shuffle)


def test(settings):
    return coco(settings, 'test', False)
