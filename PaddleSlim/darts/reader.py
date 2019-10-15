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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from PIL import ImageOps
import os
import math
import random
import tarfile
import functools
import numpy as np
from PIL import Image, ImageEnhance
import paddle
# for python2/python3 compatiablity
try:
    import cPickle
except:
    import _pickle as cPickle

IMAGE_SIZE = 32
IMAGE_DEPTH = 3
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

URL_PREFIX = 'https://www.cs.toronto.edu/~kriz/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'

paddle.dataset.common.DATA_HOME = "dataset/"

THREAD = 16
BUF_SIZE = 10240

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
IMAGENET_DIM = 224


def preprocess(sample, is_training, args):
    image_array = sample.reshape(IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE)
    rgb_array = np.transpose(image_array, (1, 2, 0))
    img = Image.fromarray(rgb_array, 'RGB')

    if is_training:
        # pad, ramdom crop, random_flip_left_right
        img = ImageOps.expand(img, (4, 4, 4, 4), fill=0)
        left_top = np.random.randint(8, size=2)
        img = img.crop((left_top[1], left_top[0], left_top[1] + IMAGE_SIZE,
                        left_top[0] + IMAGE_SIZE))
        if np.random.randint(2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(img).astype(np.float32)

    img_float = img / 255.0
    img = (img_float - CIFAR_MEAN) / CIFAR_STD

    if is_training and args.cutout:
        center = np.random.randint(IMAGE_SIZE, size=2)
        offset_width = max(0, center[0] - args.cutout_length // 2)
        offset_height = max(0, center[1] - args.cutout_length // 2)
        target_width = min(center[0] + args.cutout_length // 2, IMAGE_SIZE)
        target_height = min(center[1] + args.cutout_length // 2, IMAGE_SIZE)

        for i in range(offset_height, target_height):
            for j in range(offset_width, target_width):
                img[i][j][:] = 0.0

    img = np.transpose(img, (2, 0, 1))
    return img


def reader_generator(datasets, batch_size, is_training, is_shuffle, args):
    def read_batch(datasets, args):
        if is_shuffle:
            random.shuffle(datasets)
        for im, label in datasets:
            im = preprocess(im, is_training, args)
            yield im, [int(label)]

    def reader():
        batch_data = []
        batch_label = []
        for data in read_batch(datasets, args):
            batch_data.append(data[0])
            batch_label.append(data[1])
            if len(batch_data) == batch_size:
                batch_data = np.array(batch_data, dtype='float32')
                batch_label = np.array(batch_label, dtype='int64')
                batch_out = [batch_data, batch_label]
                yield batch_out
                batch_data = []
                batch_label = []

    return reader


def cifar10_reader(file_name, data_name, is_shuffle, args):
    with tarfile.open(file_name, mode='r') as f:
        names = [
            each_item.name for each_item in f if data_name in each_item.name
        ]
        names.sort()
        datasets = []
        for name in names:
            print("Reading file " + name)
            try:
                batch = cPickle.load(f.extractfile(name), encoding='iso-8859-1')
            except:
                batch = cPickle.load(f.extractfile(name))
            data = batch['data']
            labels = batch.get('labels', batch.get('fine_labels', None))
            assert labels is not None
            dataset = zip(data, labels)
            datasets.extend(dataset)
        if is_shuffle:
            random.shuffle(datasets)
    return datasets


def train_search(batch_size, train_portion, is_shuffle, args):
    datasets = cifar10_reader(
        paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5),
        'data_batch', is_shuffle, args)
    split_point = int(np.floor(train_portion * len(datasets)))
    train_datasets = datasets[:split_point]
    val_datasets = datasets[split_point:]
    train_readers = []
    val_readers = []
    n = int(math.ceil(len(train_datasets) // args.num_workers)
            ) if args.use_multiprocess else len(train_datasets)
    train_datasets_lists = [
        train_datasets[i:i + n] for i in range(0, len(train_datasets), n)
    ]
    val_datasets_lists = [
        val_datasets[i:i + n] for i in range(0, len(val_datasets), n)
    ]

    for pid in range(len(train_datasets_lists)):
        train_readers.append(
            reader_generator(train_datasets_lists[pid], batch_size, True, True,
                             args))
        val_readers.append(
            reader_generator(val_datasets_lists[pid], batch_size, True, True,
                             args))
    if args.use_multiprocess:
        reader = [
            paddle.reader.multiprocess_reader(train_readers, False),
            paddle.reader.multiprocess_reader(val_readers, False)
        ]
    else:
        reader = [train_readers[0], val_readers[0]]

    return reader


def train_valid(batch_size, is_train, is_shuffle, args):
    name = 'data_batch' if is_train else 'test_batch'
    datasets = cifar10_reader(
        paddle.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5), name,
        is_shuffle, args)
    n = int(math.ceil(len(datasets) // args.
                      num_workers)) if args.use_multiprocess else len(datasets)
    datasets_lists = [datasets[i:i + n] for i in range(0, len(datasets), n)]
    multi_readers = []
    for pid in range(len(datasets_lists)):
        multi_readers.append(
            reader_generator(datasets_lists[pid], batch_size, is_train,
                             is_shuffle, args))
    if args.use_multiprocess:
        reader = paddle.reader.multiprocess_reader(multi_readers, False)
    else:
        reader = multi_readers[0]
    return reader


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.BILINEAR)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


def process_image(sample, mode, color_jitter, rotate):
    img_path = sample[0]

    img = Image.open(img_path)
    if mode == 'train':
        img = random_crop(img, IMAGENET_DIM)
        if np.random.randint(0, 2) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if color_jitter:
            img = distort_color(img)

    else:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=IMAGENET_DIM, center=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= IMAGENET_MEAN
    img /= IMAGENET_STD

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=None):
    def reader():
        try:
            with open(file_list) as flist:
                full_lines = [line.strip() for line in flist]
                if shuffle:
                    np.random.shuffle(full_lines)
                lines = full_lines
                for line in lines:
                    if mode == 'train' or mode == 'val':
                        img_path, label = line.split()
                        img_path = os.path.join(data_dir, img_path)
                        yield img_path, int(label)
                    elif mode == 'test':
                        img_path = os.path.join(data_dir, line)
                        yield [img_path]
        except Exception as e:
            print("Reader failed!\n{}".format(str(e)))
            os._exit(1)

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)
    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def imagenet_reader(data_dir, mode):
    if mode is 'train':
        shuffle = True
        suffix = 'train_list.txt'
    elif mode is 'val':
        shuffle = False
        suffix = 'val_list.txt'
    file_list = os.path.join(data_dir, suffix)
    return _reader_creator(file_list, mode, shuffle, data_dir)
