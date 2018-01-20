import os
import random
import functools
import numpy as np
import paddle.v2 as paddle
from PIL import Image, ImageEnhance

random.seed(0)

_R_MEAN = 123.0
_G_MEAN = 117.0
_B_MEAN = 104.0

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 1024

DATA_DIR = 'ILSVRC2012'
TRAIN_LIST = 'ILSVRC2012/train_list.txt'
TEST_LIST = 'ILSVRC2012/test_list.txt'

img_mean = np.array([_R_MEAN, _G_MEAN, _B_MEAN]).reshape((3, 1, 1))


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
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def rotate_image(img):
    angle = random.randint(-10, 10)
    img = img.rotate(angle)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


def process_image(sample, mode):
    img_path = sample[0]

    img = Image.open(img_path)
    if mode == 'train':
        img = resize_short(img, DATA_DIM + 32)
        img = rotate_image(img)
    else:
        img = resize_short(img, DATA_DIM)
    img = crop_image(img, target_size=DATA_DIM, center=(mode != 'train'))
    if mode == 'train':
        img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1))
    img -= img_mean

    if mode == 'train' or mode == 'test':
        return img, sample[1]
    elif mode == 'infer':
        return img


def _reader_creator(file_list, mode, shuffle=False):
    def reader():
        with open(file_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                if mode == 'train' or mode == 'test':
                    img_path, label = line.split()
                    img_path = os.path.join(DATA_DIR, img_path)
                    yield img_path, int(label)
                elif mode == 'infer':
                    img_path = os.path.join(DATA_DIR, line)
                    yield [img_path]

    mapper = functools.partial(process_image, mode=mode)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def train():
    return _reader_creator(TRAIN_LIST, 'train', shuffle=True)


def test():
    return _reader_creator(TEST_LIST, 'test', shuffle=False)


def infer(file_list):
    return _reader_creator(file_list, 'infer', shuffle=False)
