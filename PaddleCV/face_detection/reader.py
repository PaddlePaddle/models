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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from PIL import ImageDraw
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy
import random
import cv2
import six
import math
from itertools import islice
import paddle
import image_util


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
        self.dataset = dataset
        self.ap_version = ap_version
        self.toy = toy
        self.data_dir = data_dir
        self.apply_distort = apply_distort
        self.apply_expand = apply_expand
        self.resize_height = resize_h
        self.resize_width = resize_w
        self.img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')
        self.expand_prob = 0.5
        self.expand_max_ratio = 4
        self.hue_prob = 0.5
        self.hue_delta = 18
        self.contrast_prob = 0.5
        self.contrast_delta = 0.5
        self.saturation_prob = 0.5
        self.saturation_delta = 0.5
        self.brightness_prob = 0.5
        # _brightness_delta is the normalized value by 256
        self.brightness_delta = 0.125
        self.scale = 0.007843  # 1 / 127.5
        self.data_anchor_sampling_prob = 0.5
        self.min_face_size = 8.0


def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image


def preprocess(img, bbox_labels, mode, settings, image_path):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    if mode == 'train':
        if settings.apply_distort:
            img = image_util.distort_image(img, settings)
        if settings.apply_expand:
            img, bbox_labels, img_width, img_height = image_util.expand_image(
                img, bbox_labels, img_width, img_height, settings)

        # sampling
        batch_sampler = []
        # used for continuous evaluation
        if 'ce_mode' in os.environ:
           random.seed(0)
           np.random.seed(0)
        prob = np.random.uniform(0., 1.)
        if prob > settings.data_anchor_sampling_prob:
            scale_array = np.array([16, 32, 64, 128, 256, 512])
            batch_sampler.append(
                image_util.sampler(1, 10, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.2,
                                   0.0, True))
            sampled_bbox = image_util.generate_batch_random_samples(
                batch_sampler, bbox_labels, img_width, img_height, scale_array,
                settings.resize_width, settings.resize_height)
            img = np.array(img)
            if len(sampled_bbox) > 0:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                img, sampled_labels = image_util.crop_image_sampling(
                    img, bbox_labels, sampled_bbox[idx], img_width, img_height,
                    settings.resize_width, settings.resize_height,
                    settings.min_face_size)

            img = img.astype('uint8')
            img = Image.fromarray(img)

        else:
            # hard-code here
            batch_sampler.append(
                image_util.sampler(1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                   0.0, True))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                   0.0, True))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                   0.0, True))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                   0.0, True))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                   0.0, True))
            sampled_bbox = image_util.generate_batch_samples(
                batch_sampler, bbox_labels, img_width, img_height)

            img = np.array(img)
            if len(sampled_bbox) > 0:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                img, sampled_labels = image_util.crop_image(
                    img, bbox_labels, sampled_bbox[idx], img_width, img_height,
                    settings.resize_width, settings.resize_height,
                    settings.min_face_size)

            img = Image.fromarray(img)
    interp_mode = [
        Image.BILINEAR, Image.HAMMING, Image.NEAREST, Image.BICUBIC,
        Image.LANCZOS
    ]
    interp_indx = np.random.randint(0, 5)

    img = img.resize(
        (settings.resize_width, settings.resize_height),
        resample=interp_mode[interp_indx])
    img = np.array(img)

    if mode == 'train':
        mirror = int(np.random.uniform(0, 2))
        if mirror == 1:
            img = img[:, ::-1, :]
            for i in six.moves.xrange(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = 1 - sampled_labels[i][3]
                sampled_labels[i][3] = 1 - tmp

    img = to_chw_bgr(img)
    img = img.astype('float32')
    img -= settings.img_mean
    img = img * settings.scale
    return img, sampled_labels


def load_file_list(input_txt):
    with open(input_txt, 'r') as f_dir:
        lines_input_txt = f_dir.readlines()

    file_dict = {}
    num_class = 0
    for i in range(len(lines_input_txt)):
        line_txt = lines_input_txt[i].strip('\n\t\r')
        if '--' in line_txt:
            if i != 0:
                num_class += 1
            file_dict[num_class] = []
            file_dict[num_class].append(line_txt)
        if '--' not in line_txt:
            if len(line_txt) > 6:
                split_str = line_txt.split(' ')
                x1_min = float(split_str[0])
                y1_min = float(split_str[1])
                x2_max = float(split_str[2])
                y2_max = float(split_str[3])
                line_txt = str(x1_min) + ' ' + str(y1_min) + ' ' + str(
                    x2_max) + ' ' + str(y2_max)
                file_dict[num_class].append(line_txt)
            else:
                file_dict[num_class].append(line_txt)

    return list(file_dict.values())


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


def train_generator(settings, file_list, batch_size, shuffle=True):
    def reader():
        if shuffle and 'ce_mode' not in os.environ:
            np.random.shuffle(file_list)
        batch_out = []
        for item in file_list:
            image_name = item[0]
            image_path = os.path.join(settings.data_dir, image_name)
            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size

            # layout: label | xmin | ymin | xmax | ymax
            bbox_labels = []
            for index_box in range(len(item)):
                if index_box >= 2:
                    bbox_sample = []
                    temp_info_box = item[index_box].split(' ')
                    xmin = float(temp_info_box[0])
                    ymin = float(temp_info_box[1])
                    w = float(temp_info_box[2])
                    h = float(temp_info_box[3])

                    # Filter out wrong labels
                    if w < 0 or h < 0:
                        continue
                    xmax = xmin + w
                    ymax = ymin + h

                    bbox_sample.append(1)
                    bbox_sample.append(float(xmin) / im_width)
                    bbox_sample.append(float(ymin) / im_height)
                    bbox_sample.append(float(xmax) / im_width)
                    bbox_sample.append(float(ymax) / im_height)
                    bbox_labels.append(bbox_sample)
            im, sample_labels = preprocess(im, bbox_labels, "train", settings,
                                           image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue

            im = im.astype('float32')
            face_box = sample_labels[:, 1:5]
            head_box = expand_bboxes(face_box)
            label = [1] * len(face_box)
            batch_out.append((im, face_box, head_box, label))
            if len(batch_out) == batch_size:
                yield batch_out
                batch_out = []

    return reader


def train(settings,
          file_list,
          batch_size,
          shuffle=True,
          use_multiprocess=True,
          num_workers=8):
    file_lists = load_file_list(file_list)
    if use_multiprocess:
        n = int(math.ceil(len(file_lists) // num_workers))
        split_lists = [
            file_lists[i:i + n] for i in range(0, len(file_lists), n)
        ]
        readers = []
        for iterm in split_lists:
            readers.append(
                train_generator(settings, iterm, batch_size, shuffle))
        return paddle.reader.multiprocess_reader(readers, False)
    else:
        return train_generator(settings, file_lists, batch_size, shuffle)


def test(settings, file_list):
    file_lists = load_file_list(file_list)

    def reader():
        for image in file_lists:
            image_name = image[0]
            image_path = os.path.join(settings.data_dir, image_name)
            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            yield im, image_path

    return reader


def infer(settings, image_path):
    def batch_reader():
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        im_width, im_height = img.size
        if settings.resize_width and settings.resize_height:
            img = img.resize((settings.resize_width, settings.resize_height),
                             Image.ANTIALIAS)
        img = np.array(img)
        img = to_chw_bgr(img)
        img = img.astype('float32')
        img -= settings.img_mean
        img = img * settings.scale
        return np.array([img])

    return batch_reader
