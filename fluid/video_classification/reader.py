import os
import sys
import math
import random
import functools
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import paddle
from PIL import Image, ImageEnhance

random.seed(0)

THREAD = 8
BUF_SIZE = 1024

TRAIN_LIST = 'data/train.list'
TEST_LIST = 'data/test.list'
INFER_LIST = 'data/test.list'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

python_ver = sys.version_info

def imageloader(buf):
    if isinstance(buf, str):
        img = Image.open(StringIO(buf))
    else:
        img = Image.open(BytesIO(buf))

    return img.convert('RGB')


def group_scale(imgs, target_size):
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images


def group_random_flip(img_group):
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group


def group_center_crop(img_group, target_size):
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


def video_loader(frames, nsample, mode):
    videolen = len(frames)
    average_dur = videolen // nsample

    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= 1:
                idx = random.randint(0, average_dur - 1)
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= 1:
                idx = (average_dur - 1) // 2
                idx += i * average_dur
            else:
                idx = i

        imgbuf = frames[int(idx % videolen)]
        img = imageloader(imgbuf)
        imgs.append(img)

    return imgs


def decode_pickle(sample, mode, seg_num, short_size, target_size):
    pickle_path = sample[0]
    if python_ver < (3, 0):
        data_loaded = pickle.load(open(pickle_path, 'rb'))
    else:
        data_loaded = pickle.load(open(pickle_path, 'rb'), encoding='bytes')
    vid, label, frames = data_loaded

    imgs = video_loader(frames, seg_num, mode)
    imgs = group_scale(imgs, short_size)

    if mode == 'train':
        imgs = group_random_crop(imgs, target_size)
        imgs = group_random_flip(imgs)
    else:
        imgs = group_center_crop(imgs, target_size)

    np_imgs = (np.array(imgs[0]).astype('float32').transpose(
        (2, 0, 1))).reshape(1, 3, 224, 224) / 255
    for i in range(len(imgs) - 1):
        img = (np.array(imgs[i + 1]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, 224, 224) / 255
        np_imgs = np.concatenate((np_imgs, img))
    imgs = np_imgs
    imgs -= img_mean
    imgs /= img_std

    if mode == 'train' or mode == 'test':
        return imgs, label
    elif mode == 'infer':
        return imgs, vid


def _reader_creator(pickle_list,
                    mode,
                    seg_num,
                    short_size,
                    target_size,
                    shuffle=False):
    def reader():
        with open(pickle_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                pickle_path = line.strip()
                yield [pickle_path]

    mapper = functools.partial(
        decode_pickle,
        mode=mode,
        seg_num=seg_num,
        short_size=short_size,
        target_size=target_size)

    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def train(seg_num):
    return _reader_creator(
        TRAIN_LIST,
        'train',
        shuffle=True,
        seg_num=seg_num,
        short_size=256,
        target_size=224)


def test(seg_num):
    return _reader_creator(
        TEST_LIST,
        'test',
        shuffle=False,
        seg_num=seg_num,
        short_size=256,
        target_size=224)


def infer(seg_num):
    return _reader_creator(
        INFER_LIST,
        'infer',
        shuffle=False,
        seg_num=seg_num,
        short_size=256,
        target_size=224)
