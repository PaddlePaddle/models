import os
import math
import random
import functools
import numpy as np
import cv2
import io

import paddle
import paddle.fluid as fluid

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 2048


img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    resized = cv2.resize(
        img,
        (resized_width, resized_height),
    )
    return resized


def crop_image(img, target_size, center):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img

def process_image(sample,
                  settings,
                  mode,
                  color_jitter,
                  rotate,
                  crop_size=224,
                  mean=None,
                  std=None):
    """ process_image """

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std

    img_path = sample
    img = cv2.imread(img_path)

    if crop_size > 0:
        target_size = settings.resize_short_size
        img = resize_short(img, target_size)

        img = crop_image(img, target_size=crop_size, center=True)

    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return (img, )


def image_mapper(**kwargs):
    """ image_mapper """
    return functools.partial(process_image, **kwargs)


def process_batch_data(input_data, settings, mode, color_jitter, rotate):
    batch_data = []
    for sample in input_data:
        batch_data.append(
            process_image(sample, settings, mode, color_jitter, rotate))
    return batch_data


def _reader_creator(settings,
                    file_list,
                    batch_size,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    data_dir=None,
                    shuffle_seed=0):
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
                    yield [sample[0] for sample in batch_data]
                    batch_data = []

        return read_file_list

    data_reader = reader()

    mapper = functools.partial(
        process_batch_data,
        settings=settings,
        mode=mode,
        color_jitter=color_jitter,
        rotate=rotate)

    return paddle.reader.xmap_readers(
        mapper, data_reader, THREAD, BUF_SIZE, order=False)

def test(settings, batch_size=1, data_dir=None):
    file_list = settings.img_list
    data_dir = settings.img_path
    return _reader_creator(
        settings,
        file_list,
        batch_size,
        'test',
        shuffle=False,
        data_dir=data_dir)
