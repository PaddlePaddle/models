# Copyright (c) 2019 PaddlePaddle Authors. All Rig hts Reserved
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
# DARTS
# Copyright (c) 2018, Hanxiao Liu.
# Licensed under the Apache License, Version 2.0;
# --------------------------------------------------------
"""
CIFAR-10 dataset.
This module will download dataset from
https://www.cs.toronto.edu/~kriz/cifar.html and parse train/test set into
paddle reader creators.
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
with 6000 images per class. There are 50000 training images and 10000 test images.
"""

from PIL import Image
from PIL import ImageOps
import numpy as np

import cPickle
import random
import utils
import paddle.fluid as fluid
import time
import os
import functools
import paddle.reader

__all__ = ['train10', 'test10']

image_size = 32
image_depth = 3
half_length = 8

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]


def generate_reshape_label(label, batch_size, CIFAR_CLASSES=10):
    reshape_label = np.zeros((batch_size, 1), dtype='int32')
    reshape_non_label = np.zeros(
        (batch_size * (CIFAR_CLASSES - 1), 1), dtype='int32')
    num = 0
    for i in range(batch_size):
        label_i = label[i]
        reshape_label[i] = label_i + i * CIFAR_CLASSES
        for j in range(CIFAR_CLASSES):
            if label_i != j:
                reshape_non_label[num] = \
                          j + i * CIFAR_CLASSES
                num += 1
    return reshape_label, reshape_non_label


def generate_bernoulli_number(batch_size, CIFAR_CLASSES=10):
    rcc_iters = 50
    rad_var = np.zeros((rcc_iters, batch_size, CIFAR_CLASSES - 1))
    for i in range(rcc_iters):
        bernoulli_num = np.random.binomial(size=batch_size, n=1, p=0.5)
        bernoulli_map = np.array([])
        ones = np.ones((CIFAR_CLASSES - 1, 1))
        for batch_id in range(batch_size):
            num = bernoulli_num[batch_id]
            var_id = 2 * ones * num - 1
            bernoulli_map = np.append(bernoulli_map, var_id)
        rad_var[i] = bernoulli_map.reshape((batch_size, CIFAR_CLASSES - 1))
    return rad_var.astype('float32')


def preprocess(sample, is_training, args):
    image_array = sample.reshape(3, image_size, image_size)
    rgb_array = np.transpose(image_array, (1, 2, 0))
    img = Image.fromarray(rgb_array, 'RGB')

    if is_training:
        # pad and ramdom crop
        img = ImageOps.expand(img, (4, 4, 4, 4), fill=0)  # pad to 40 * 40 * 3
        left_top = np.random.randint(9, size=2)  # rand 0 - 8
        img = img.crop((left_top[0], left_top[1], left_top[0] + image_size,
                        left_top[1] + image_size))
        if np.random.randint(2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.array(img).astype(np.float32)

    # per_image_standardization
    img_float = img / 255.0
    img = (img_float - CIFAR_MEAN) / CIFAR_STD

    if is_training and args.cutout:
        center = np.random.randint(image_size, size=2)
        offset_width = max(0, center[0] - half_length)
        offset_height = max(0, center[1] - half_length)
        target_width = min(center[0] + half_length, image_size)
        target_height = min(center[1] + half_length, image_size)

        for i in range(offset_height, target_height):
            for j in range(offset_width, target_width):
                img[i][j][:] = 0.0

    img = np.transpose(img, (2, 0, 1))
    return img


def reader_creator_filepath(filename, sub_name, is_training, args):
    files = os.listdir(filename)
    names = [each_item for each_item in files if sub_name in each_item]
    names.sort()
    datasets = []
    for name in names:
        print("Reading file " + name)
        batch = cPickle.load(open(filename + name, 'rb'))
        data = batch['data']
        labels = batch.get('labels', batch.get('fine_labels', None))
        assert labels is not None
        dataset = zip(data, labels)
        datasets.extend(dataset)
    random.shuffle(datasets)

    def read_batch(datasets, args):
        for sample, label in datasets:
            im = preprocess(sample, is_training, args)
            yield im, [int(label)]

    def reader():
        batch_data = []
        batch_label = []
        for data, label in read_batch(datasets, args):
            batch_data.append(data)
            batch_label.append(label)
            if len(batch_data) == args.batch_size:
                batch_data = np.array(batch_data, dtype='float32')
                batch_label = np.array(batch_label, dtype='int64')
                if is_training:
                    flatten_label, flatten_non_label = \
                      generate_reshape_label(batch_label, args.batch_size)
                    rad_var = generate_bernoulli_number(args.batch_size)
                    mixed_x, y_a, y_b, lam = utils.mixup_data(
                        batch_data, batch_label, args.batch_size,
                        args.mix_alpha)
                    batch_out = [[mixed_x, y_a, y_b, lam, flatten_label, \
                                flatten_non_label, rad_var]]
                    yield batch_out
                else:
                    batch_out = [[batch_data, batch_label]]
                    yield batch_out
                batch_data = []
                batch_label = []

    return reader


def train10(args):
    """
    CIFAR-10 training set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Training reader creator
    :rtype: callable
    """

    return reader_creator_filepath(args.data, 'data_batch', True, args)


def test10(args):
    """
    CIFAR-10 test set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator_filepath(args.data, 'test_batch', False, args)
