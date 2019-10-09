#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import math
import random
import functools
import numpy as np
from PIL import Image, ImageEnhance

from paddle import fluid

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 2048

DATA_DIR = 'data/ILSVRC2012'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


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
    img = img.resize((size, size), Image.LANCZOS)
    return img


def rotate_image(img):
    angle = np.random.randint(-10, 11)
    img = img.rotate(angle)
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
        if rotate: img = rotate_image(img)
        img = random_crop(img, DATA_DIM)
    else:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=DATA_DIM, center=True)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if np.random.randint(0, 2) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]


def process_batch_data(input_data, mode, color_jitter, rotate):
    batch_data = []
    for sample in input_data:
        batch_data.append(process_image(sample, mode, color_jitter, rotate))
    return batch_data


def _reader_creator(file_list,
                    batch_size,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=DATA_DIR,
                    shuffle_seed=0,
                    infinite=False):
    def reader():
        def read_file_list():
            with open(file_list) as flist:
                full_lines = [line.strip() for line in flist]
                if shuffle:
                    if shuffle_seed is not None:
                        np.random.seed(shuffle_seed)
                    np.random.shuffle(full_lines)
            batch_data = []
            for line in full_lines:
                img_path, label = line.split()
                img_path = os.path.join(data_dir, img_path)
                batch_data.append([img_path, int(label)])
                if len(batch_data) == batch_size:
                    if mode == 'train' or mode == 'val':
                        yield batch_data
                    elif mode == 'test':
                        yield [sample[0] for sample in batch_data]
                    batch_data = []

        return read_file_list

    data_reader = reader()
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    if mode == 'train' and num_trainers > 1:
        assert shuffle_seed is not None, \
            "If num_trainers > 1, the shuffle_seed must be set, because " \
            "the order of batch data generated by reader " \
            "must be the same in the respective processes."
        data_reader = fluid.contrib.reader.distributed_batch_reader(data_reader)

    mapper = functools.partial(
        process_batch_data, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return fluid.io.xmap_readers(mapper, data_reader, THREAD, BUF_SIZE)


def train(batch_size, data_dir=DATA_DIR, shuffle_seed=0, infinite=False):
    file_list = os.path.join(data_dir, 'train_list.txt')
    return _reader_creator(
        file_list,
        batch_size,
        'train',
        shuffle=True,
        color_jitter=False,
        rotate=False,
        data_dir=data_dir,
        shuffle_seed=shuffle_seed,
        infinite=infinite)


def val(batch_size, data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val_list.txt')
    return _reader_creator(
        file_list, batch_size, 'val', shuffle=False, data_dir=data_dir)


def test(batch_size, data_dir=DATA_DIR):
    file_list = os.path.join(data_dir, 'val_list.txt')
    return _reader_creator(
        file_list, batch_size, 'test', shuffle=False, data_dir=data_dir)
