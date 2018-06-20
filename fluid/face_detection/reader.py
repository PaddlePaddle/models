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

import image_util
from paddle.utils.image_util import *
import random
from PIL import Image
from PIL import ImageDraw
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy


class Settings(object):
    def __init__(self,
                 dataset=None,
                 data_dir=None,
                 label_file=None,
                 resize_h=None,
                 resize_w=None,
                 mean_value=[104., 117., 123.],
                 apply_distort=True,
                 apply_expand=True,
                 ap_version='11point',
                 toy=0):
        self._dataset = dataset
        self._ap_version = ap_version
        self._toy = toy
        self._data_dir = data_dir
        self._apply_distort = apply_distort
        self._apply_expand = apply_expand
        self._resize_height = resize_h
        self._resize_width = resize_w
        self._img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')
        self._expand_prob = 0.5
        self._expand_max_ratio = 4
        self._hue_prob = 0.5
        self._hue_delta = 18
        self._contrast_prob = 0.5
        self._contrast_delta = 0.5
        self._saturation_prob = 0.5
        self._saturation_delta = 0.5
        self._brightness_prob = 0.5
        # _brightness_delta is the normalized value by 256
        # self._brightness_delta = 32
        self._brightness_delta = 0.125

    @property
    def dataset(self):
        return self._dataset

    @property
    def ap_version(self):
        return self._ap_version

    @property
    def toy(self):
        return self._toy

    @property
    def apply_expand(self):
        return self._apply_expand

    @property
    def apply_distort(self):
        return self._apply_distort

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        self._data_dir = data_dir

    @property
    def label_list(self):
        return self._label_list

    @property
    def resize_h(self):
        return self._resize_height

    @property
    def resize_w(self):
        return self._resize_width

    @property
    def img_mean(self):
        return self._img_mean


def preprocess(img, bbox_labels, mode, settings):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    if mode == 'train':
        if settings._apply_distort:
            img = image_util.distort_image(img, settings)
        if settings._apply_expand:
            img, bbox_labels, img_width, img_height = image_util.expand_image(
                img, bbox_labels, img_width, img_height, settings)
        # sampling
        batch_sampler = []
        # hard-code here
        batch_sampler.append(
            image_util.sampler(1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                               True))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                               True))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                               True))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                               True))
        batch_sampler.append(
            image_util.sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                               True))
        sampled_bbox = image_util.generate_batch_samples(
            batch_sampler, bbox_labels, img_width, img_height)

        img = np.array(img)
        if len(sampled_bbox) > 0:
            idx = int(random.uniform(0, len(sampled_bbox)))
            img, sampled_labels = image_util.crop_image(
                img, bbox_labels, sampled_bbox[idx], img_width, img_height)

        img = Image.fromarray(img)
    img = img.resize((settings.resize_w, settings.resize_h), Image.ANTIALIAS)
    img = np.array(img)

    if mode == 'train':
        mirror = int(random.uniform(0, 2))
        if mirror == 1:
            img = img[:, ::-1, :]
            for i in xrange(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = 1 - sampled_labels[i][3]
                sampled_labels[i][3] = 1 - tmp
    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img[[2, 1, 0], :, :]
    img = img.astype('float32')
    img -= settings.img_mean
    img = img * 0.007843
    return img, sampled_labels


def put_txt_in_dict(input_txt):
    with open(input_txt, 'r') as f_dir:
        lines_input_txt = f_dir.readlines()

    dict_input_txt = {}
    num_class = 0
    for i in range(len(lines_input_txt)):
        tmp_line_txt = lines_input_txt[i].strip('\n\t\r')
        if '--' in tmp_line_txt:
            if i != 0:
                num_class += 1
            dict_input_txt[num_class] = []
            dict_name = tmp_line_txt
            dict_input_txt[num_class].append(tmp_line_txt)
        if '--' not in tmp_line_txt:
            if len(tmp_line_txt) > 6:
                # tmp_line_txt = tmp_line_txt[:-2]
                split_str = tmp_line_txt.split(' ')
                x1_min = float(split_str[0])
                y1_min = float(split_str[1])
                x2_max = float(split_str[2])
                y2_max = float(split_str[3])
                tmp_line_txt = str(x1_min) + ' ' + str(y1_min) + ' ' + str(
                    x2_max) + ' ' + str(y2_max)
                dict_input_txt[num_class].append(tmp_line_txt)
            else:
                dict_input_txt[num_class].append(tmp_line_txt)

    return dict_input_txt


def expand_bboxes(bboxes,
                  expand_left=2.,
                  expand_up=2.,
                  expand_right=2.,
                  expand_down=2.):
    """
    Expand bboxes, expand 2 times by defalut.
    """
    expand_boxes = []
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        w = xmax - xmin
        h = ymax - ymin
        ex_xmin = max(xmin - w / expand_left, 0.)
        ex_ymin = max(ymin - h / expand_up, 0.)
        ex_xmax = min(xmax + w / expand_right, 1.)
        ex_ymax = min(ymax + h / expand_down, 1.)
        expand_boxes.append([ex_xmin, ex_ymin, ex_xmax, ex_ymax])
    return expand_boxes


def pyramidbox(settings, file_list, mode, shuffle):

    dict_input_txt = {}
    dict_input_txt = put_txt_in_dict(file_list)

    def reader():
        if mode == 'train' and shuffle:
            random.shuffle(dict_input_txt)
        for index_image in range(len(dict_input_txt)):

            image_name = dict_input_txt[index_image][0] + '.jpg'
            image_path = os.path.join(settings.data_dir, image_name)

            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size

            # layout: label | xmin | ymin | xmax | ymax
            if mode == 'train':
                bbox_labels = []
                for index_box in range(len(dict_input_txt[index_image])):
                    if index_box >= 2:
                        bbox_sample = []
                        temp_info_box = dict_input_txt[index_image][
                            index_box].split(' ')
                        xmin = float(temp_info_box[0])
                        ymin = float(temp_info_box[1])
                        w = float(temp_info_box[2])
                        h = float(temp_info_box[3])
                        xmax = xmin + w
                        ymax = ymin + h

                        bbox_sample.append(1)
                        bbox_sample.append(float(xmin) / im_width)
                        bbox_sample.append(float(ymin) / im_height)
                        bbox_sample.append(float(xmax) / im_width)
                        bbox_sample.append(float(ymax) / im_height)
                        bbox_labels.append(bbox_sample)

                im, sample_labels = preprocess(im, bbox_labels, mode, settings)
                sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0: continue
                im = im.astype('float32')
                boxes = sample_labels[:, 1:5]
                lbls = [1] * len(boxes)
                difficults = [1] * len(boxes)
                yield im, boxes, expand_bboxes(boxes), lbls, difficults

            if mode == 'test':
                if settings.resize_w and settings.resize_h:
                    im = im.resize((settings.resize_w, settings.resize_h),
                                   Image.ANTIALIAS)
                yield im, image_path

    return reader


def train(settings, file_list, shuffle=True):
    return pyramidbox(settings, file_list, 'train', shuffle)


def test(settings, file_list):
    return pyramidbox(settings, file_list, 'test', False)


def infer(settings, image_path):
    def batch_reader():
        img = Image.open(image_path)
        if img.mode == 'L':
            img = im.convert('RGB')
        im_width, im_height = img.size
        if settings.resize_w and settings.resize_h:
            img = img.resize((settings.resize_w, settings.resize_h),
                             Image.ANTIALIAS)
        img = np.array(img)
        # HWC to CHW
        if len(img.shape) == 3:
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 1, 0)
        # RBG to BGR
        img = img[[2, 1, 0], :, :]
        img = img.astype('float32')
        img -= settings.img_mean
        img = img * 0.007843
        img = [img]
        img = np.array(img)
        return img

    return batch_reader
