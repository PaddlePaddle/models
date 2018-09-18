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
import itertools
import paddle.dataset.common
import tarfile
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("random_flip_left_right", True,
                     "random flip left and right")
flags.DEFINE_boolean("random_flip_up_down", False, "random flip up and down")
flags.DEFINE_boolean("cutout", True, "cutout")
flags.DEFINE_boolean("standardize_image", True, "standardize input images")
flags.DEFINE_boolean("pad_and_cut_image", True, "pad and cut input images")

__all__ = ['train10', 'test10', 'convert']

URL_PREFIX = 'https://www.cs.toronto.edu/~kriz/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'

paddle.dataset.common.DATA_HOME = "dataset/"

image_size = 32
image_depth = 3
half_length = 8


def preprocess(sample, is_training):
    image_array = sample.reshape(3, image_size, image_size)
    rgb_array = np.transpose(image_array, (1, 2, 0))
    img = Image.fromarray(rgb_array, 'RGB')

    if is_training:
        if FLAGS.pad_and_cut_image:
            # pad and ramdom crop
            img = ImageOps.expand(
                img, (2, 2, 2, 2), fill=0)  # pad to 36 * 36 * 3
            left_top = np.random.randint(5, size=2)  # rand 0 - 4
            img = img.crop((left_top[0], left_top[1], left_top[0] + image_size,
                            left_top[1] + image_size))

        if FLAGS.random_flip_left_right and np.random.randint(2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if FLAGS.random_flip_up_down and np.random.randint(2):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

    img = np.array(img).astype(np.float32)

    if FLAGS.standardize_image:
        # per_image_standardization
        img_float = img / 255.0
        mean = np.mean(img_float)
        std = max(np.std(img_float), 1.0 / np.sqrt(3 * image_size * image_size))
        img = (img_float - mean) / std

    if is_training and FLAGS.cutout:
        center = np.random.randint(image_size, size=2)
        offset_width = max(0, center[0] - half_length)
        offset_height = max(0, center[1] - half_length)
        target_width = min(center[0] + half_length, image_size)
        target_height = min(center[1] + half_length, image_size)

        for i in range(offset_height, target_height):
            for j in range(offset_width, target_width):
                img[i][j][:] = 0.0

    img = np.transpose(img, (2, 0, 1))
    return img.reshape(3 * image_size * image_size)


def reader_creator(filename, sub_name, is_training):
    def read_batch(batch):
        data = batch['data']
        labels = batch.get('labels', batch.get('fine_labels', None))
        assert labels is not None
        for sample, label in itertools.izip(data, labels):
            yield preprocess(sample, is_training), int(label)

    def reader():
        with tarfile.open(filename, mode='r') as f:
            names = [
                each_item.name for each_item in f if sub_name in each_item.name
            ]
            names.sort()

            for name in names:
                print("Reading file " + name)
                batch = cPickle.load(f.extractfile(name))
                for item in read_batch(batch):
                    yield item

    return reader


def train10():
    """
    CIFAR-10 training set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5),
        'data_batch', True)


def test10():
    """
    CIFAR-10 test set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5),
        'test_batch', False)


def fetch():
    paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5)


def convert(path):
    """
    Converts dataset to recordio format
    """
    paddle.dataset.common.convert(path, train10(), 1000, "cifar_train10")
    paddle.dataset.common.convert(path, test10(), 1000, "cifar_test10")
