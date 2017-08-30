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
import numpy as np
import xml.etree.ElementTree
import os


class Settings(object):
    def __init__(self, data_dir, label_file, resize_h, resize_w, mean_value):
        self._data_dir = data_dir
        self._label_list = []
        label_fpath = os.path.join(data_dir, label_file)
        for line in open(label_fpath):
            self._label_list.append(line.strip())

        self._resize_height = resize_h
        self._resize_width = resize_w
        self._img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')

    @property
    def data_dir(self):
        return self._data_dir

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


def _reader_creator(settings, file_list, mode, shuffle):
    def reader():
        with open(file_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                if mode == 'train' or mode == 'test':
                    img_path, label_path = line.split()
                    img_path = os.path.join(settings.data_dir, img_path)
                    label_path = os.path.join(settings.data_dir, label_path)
                elif mode == 'infer':
                    img_path = os.path.join(settings.data_dir, line)

                img = Image.open(img_path)
                img_width, img_height = img.size
                img = np.array(img)

                # layout: label | xmin | ymin | xmax | ymax | difficult
                if mode == 'train' or mode == 'test':
                    bbox_labels = []
                    root = xml.etree.ElementTree.parse(label_path).getroot()
                    for object in root.findall('object'):
                        bbox_sample = []
                        # start from 1
                        bbox_sample.append(
                            float(
                                settings.label_list.index(
                                    object.find('name').text)))
                        bbox = object.find('bndbox')
                        difficult = float(object.find('difficult').text)
                        bbox_sample.append(
                            float(bbox.find('xmin').text) / img_width)
                        bbox_sample.append(
                            float(bbox.find('ymin').text) / img_height)
                        bbox_sample.append(
                            float(bbox.find('xmax').text) / img_width)
                        bbox_sample.append(
                            float(bbox.find('ymax').text) / img_height)
                        bbox_sample.append(difficult)
                        bbox_labels.append(bbox_sample)

                    sample_labels = bbox_labels
                    if mode == 'train':
                        batch_sampler = []
                        # hard-code here
                        batch_sampler.append(
                            image_util.sampler(1, 1, 1.0, 1.0, 1.0, 1.0, 0.0,
                                               0.0))
                        batch_sampler.append(
                            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.1,
                                               0.0))
                        batch_sampler.append(
                            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.3,
                                               0.0))
                        batch_sampler.append(
                            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.5,
                                               0.0))
                        batch_sampler.append(
                            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.7,
                                               0.0))
                        batch_sampler.append(
                            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.9,
                                               0.0))
                        batch_sampler.append(
                            image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.0,
                                               1.0))
                        """ random crop """
                        sampled_bbox = image_util.generate_batch_samples(
                            batch_sampler, bbox_labels, img_width, img_height)

                        if len(sampled_bbox) > 0:
                            idx = int(random.uniform(0, len(sampled_bbox)))
                            img, sample_labels = image_util.crop_image(
                                img, bbox_labels, sampled_bbox[idx], img_width,
                                img_height)

                img = Image.fromarray(img)
                img = img.resize((settings.resize_w, settings.resize_h),
                                 Image.ANTIALIAS)
                img = np.array(img)

                if mode == 'train':
                    mirror = int(random.uniform(0, 2))
                    if mirror == 1:
                        img = img[:, ::-1, :]
                        for i in xrange(len(sample_labels)):
                            tmp = sample_labels[i][1]
                            sample_labels[i][1] = 1 - sample_labels[i][3]
                            sample_labels[i][3] = 1 - tmp

                if len(img.shape) == 3:
                    img = np.swapaxes(img, 1, 2)
                    img = np.swapaxes(img, 1, 0)

                img = img.astype('float32')
                img -= settings.img_mean
                img = img.flatten()

                if mode == 'train' or mode == 'test':
                    if mode == 'train' and len(sample_labels) == 0: continue
                    yield img.astype('float32'), sample_labels
                elif mode == 'infer':
                    yield img.astype('float32')

    return reader


def train(settings, file_list, shuffle=True):
    return _reader_creator(settings, file_list, 'train', shuffle)


def test(settings, file_list):
    return _reader_creator(settings, file_list, 'test', False)


def infer(settings, file_list):
    return _reader_creator(settings, file_list, 'infer', False)
