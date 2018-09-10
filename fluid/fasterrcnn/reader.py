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
            self.train_file_list = 'annotations/instances_train2017.json'
            self.train_data_dir = 'train2017'
            self.val_file_list = 'annotations/instances_val2017.json'
            self.val_data_dir = 'val2017'
        elif 'pascalvoc' in args.dataset:
            self.label_file = 'label_list'
            self.train_file_list = 'trainval.txt'
            self.val_file_list = 'test.txt'
            self.label_list = []
            label_fpath = os.path.join(self.data_dir, self.label_file)
            for line in open(label_fpath):
                self.label_list.append(line.strip())
        else:
            raise NotImplementedError('Dataset {} not supported'.format(
                self.dataset))
        self.img_mean = np.array(self.mean_value)[:, np.newaxis,
                                                  np.newaxis].astype('float32')


def preprocess(img, bbox_labels, mode, settings):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    img = img.resize((settings.resize_w, settings.resize_h), Image.ANTIALIAS)
    img = np.array(img)

    if mode == 'train':
        flip = int(np.random.uniform())
        if flip == 1:
            img = img[:, :, ::-1]
            for i in six.moves.xrange(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = img_height - sampled_labels[i][3]
                sampled_labels[i][3] = img_height - tmp
    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img.astype('float32')
    img -= settings.img_mean
    return img, sampled_labels


def coco(settings, file_list, mode, shuffle):
    if mode == 'train':
        settings.file_list = settings.train_file_list
        settings.data_dir = os.path.join(settings.data_dir,
                                         settings.train_data_dir)
    elif mode == 'test':
        settings.file_list = settings.train_file_list
        settings.data_dir = os.path.join(settings.data_dir,
                                         settings.train_data_dir)

    json_dataset = JsonDataset(settings)
    roidbs = json_dataset.get_roidb(train=(mode == 'train'))

    print("{} on {} with {} roidbs".format(mode, settings.dataset, len(roidbs)))

    def reader():
        if mode == "train" and shuffle:
            random.shuffle(roidbs)
        for roidb in roidbs:
            im, im_scales = data_utils.get_image_blob(roidb, settings)
            im_height = np.round(roidb['height'] * im_scales)
            im_width = np.round(roidb['width'] * im_scales)
            im_info = np.array(
                [[im_height, im_width, im_scales]], dtype=np.float32)
            gt_boxes = roidb['gt_boxes'].astype('float32')
            gt_classes = roidb['gt_classes'].astype('int32')
            is_crowd = roidb['is_crowd'].astype('int32')
            yield im, gt_boxes, gt_classes, is_crowd, im_info

    return reader


def pascalvoc(settings, file_list, mode, shuffle):
    flist = open(file_list)
    images = [line.strip() for line in flist]
    print("{} on {} with {} images".format(mode, settings.dataset, len(images)))

    def reader():
        if mode == "train" and shuffle:
            random.shuffle(images)
        for image in images:
            image_path, label_path = image.split()
            image_path = os.path.join(settings.data_dir, image_path)
            label_path = os.path.join(settings.data_dir, label_path)

            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size
            im_info = [float(im_width), float(im_height), 16.0]
            # layout: label | xmin | ymin | xmax | ymax | im_info
            bbox_labels = []
            root = xml.etree.ElementTree.parse(label_path).getroot()
            for object in root.findall('object'):
                bbox_sample = []
                # start from 1
                bbox_sample.append(
                    float(settings.label_list.index(object.find('name').text)))
                bbox = object.find('bndbox')
                bbox_sample.append(float(bbox.find('xmin').text))
                bbox_sample.append(float(bbox.find('ymin').text))
                bbox_sample.append(float(bbox.find('xmax').text))
                bbox_sample.append(float(bbox.find('ymax').text))
                bbox_labels.append(bbox_sample)
            im, sample_labels = preprocess(im, bbox_labels, mode, settings)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue
            im = im.astype('float32')
            boxes = sample_labels[:, 1:5]
            lbls = sample_labels[:, 0].astype('int32')
            is_crowd = np.zeros(len(boxes), dtype='int32')
            yield im, boxes, lbls, is_crowd, im_info

    return reader


def train(settings, file_list, shuffle=True):
    file_list = os.path.join(settings.data_dir, file_list)
    if 'coco' in settings.dataset:
        return coco(settings, 'train', shuffle)
    else:
        return pascalvoc(settings, file_list, 'train', shuffle)


def test(settings, file_list):
    file_list = os.path.join(settings.data_dir, file_list)
    if 'coco' in settings.dataset:
        return coco(settings, 'test', False)
    else:
        return pascalvoc(settings, file_list, 'test', False)


def infer(settings, image_path):
    def reader():
        img = Image.open(image_path)
        if img.mode == 'L':
            img = im.convert('RGB')
        im_width, im_height = img.size
        img = img.resize((settings.resize_w, settings.resize_h),
                         Image.ANTIALIAS)
        img = np.array(img)
        # HWC to CHW
        if len(img.shape) == 3:
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 1, 0)
        img -= settings.img_mean
        return img

    return reader
