import os
import math
import random
import functools
import numpy as np
import paddle
import cv2
import io

random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 102400

DATA_DIR = 'data/ILSVRC2012'

TRAIN_LIST = 'data/ILSVRC2012/train_list.txt'
TEST_LIST = 'data/ILSVRC2012/val_list.txt'

#random.seed(0)


def rotate_image(img):
    """ rotate_image """
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = random.randint(-10, 10)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def random_crop(img, size, scale=None, ratio=None):
    """ random_crop """
    scale = [0.08, 1.0] if scale is None else scale
    ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

    aspect_ratio = math.sqrt(random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[1]) / img.shape[0]) / (w**2),
                (float(img.shape[0]) / img.shape[1]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * random.uniform(scale_min,
                                                               scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = random.randint(0, img.shape[0] - h)
    j = random.randint(0, img.shape[1] - w)

    img = img[i:i + h, j:j + w, :]
    resized = cv2.resize(img, (size, size))
    return resized


def distort_color(img):
    return img


def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def process_image(sample,
                  mode,
                  color_jitter,
                  rotate,
                  crop_size=224,
                  mean=None,
                  std=None):
    """ process_image """

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std

    img_path = sample[0]
    img = cv2.imread(img_path)

    if mode == 'train':
        if rotate:
            img = rotate_image(img)
        if crop_size > 0:
            img = random_crop(img, crop_size)
        if color_jitter:
            img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = img[:, ::-1, :]
    else:
        if crop_size > 0:
            img = resize_short(img, crop_size)
            img = crop_image(img, target_size=crop_size, center=True)

    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return (img, sample[1])
    elif mode == 'test':
        return (img, )


def image_mapper(**kwargs):
    """ image_mapper """
    return functools.partial(process_image, **kwargs)


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False):
    def reader():
        with open(file_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                if mode == 'train' or mode == 'val':
                    img_path, label = line.split()
                    img_path = os.path.join(DATA_DIR, img_path)
                    yield img_path, int(label)
                elif mode == 'test':
                    img_path = os.path.join(DATA_DIR, line)
                    yield [img_path]

    image_mapper = functools.partial(
        process_image,
        mode=mode,
        color_jitter=color_jitter,
        rotate=color_jitter,
        crop_size=224)
    reader = paddle.reader.xmap_readers(
        image_mapper, reader, THREAD, BUF_SIZE, order=False)
    return reader


def train(file_list=TRAIN_LIST):
    return _reader_creator(
        file_list, 'train', shuffle=True, color_jitter=False, rotate=False)


def val(file_list=TEST_LIST):
    return _reader_creator(file_list, 'val', shuffle=False)


def test(file_list):
    return _reader_creator(file_list, 'test', shuffle=False)
