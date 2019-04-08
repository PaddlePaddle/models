#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import logging

from .reader_utils import DataReader

logger = logging.getLogger(__name__)
python_ver = sys.version_info


class KineticsReader(DataReader):
    """
    Data reader for kinetics dataset of two format mp4 and pkl.
    1. mp4, the original format of kinetics400
    2. pkl, the mp4 was decoded previously and stored as pkl
    In both case, load the data, and then get the frame data in the form of numpy and label as an integer.
     dataset cfg: format
                  num_classes
                  seg_num
                  short_size
                  target_size
                  num_reader_threads
                  buf_size
                  image_mean
                  image_std
                  batch_size
                  list
    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.format = cfg.MODEL.format
        self.num_classes = cfg.MODEL.num_classes
        self.seg_num = cfg.MODEL.seg_num
        self.seglen = cfg.MODEL.seglen
        self.short_size = cfg[mode.upper()]['short_size']
        self.target_size = cfg[mode.upper()]['target_size']
        self.num_reader_threads = cfg[mode.upper()]['num_reader_threads']
        self.buf_size = cfg[mode.upper()]['buf_size']

        self.img_mean = np.array(cfg.MODEL.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape(
            [3, 1, 1]).astype(np.float32)
        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']

    def create_reader(self):
        _reader = _reader_creator(self.filelist, self.mode, seg_num=self.seg_num, seglen = self.seglen, \
                             short_size = self.short_size, target_size = self.target_size, \
                             img_mean = self.img_mean, img_std = self.img_std, \
                             shuffle = (self.mode == 'train'), \
                             num_threads = self.num_reader_threads, \
                             buf_size = self.buf_size, format = self.format)

        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []

        return _batch_reader


def _reader_creator(pickle_list,
                    mode,
                    seg_num,
                    seglen,
                    short_size,
                    target_size,
                    img_mean,
                    img_std,
                    shuffle=False,
                    num_threads=1,
                    buf_size=1024,
                    format='pkl'):
    def reader():
        with open(pickle_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                pickle_path = line.strip()
                yield [pickle_path]

    if format == 'pkl':
        decode_func = decode_pickle
    elif format == 'mp4':
        decode_func = decode_mp4
    else:
        raise "Not implemented format {}".format(format)

    mapper = functools.partial(
        decode_func,
        mode=mode,
        seg_num=seg_num,
        seglen=seglen,
        short_size=short_size,
        target_size=target_size,
        img_mean=img_mean,
        img_std=img_std)

    return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size)


def decode_mp4(sample, mode, seg_num, seglen, short_size, target_size, img_mean,
               img_std):
    sample = sample[0].split(' ')
    mp4_path = sample[0]
    # when infer, we store vid as label
    label = int(sample[1])
    try:
        imgs = mp4_loader(mp4_path, seg_num, seglen, mode)
        if len(imgs) < 1:
            logger.error('{} frame length {} less than 1.'.format(mp4_path,
                                                                  len(imgs)))
            return None, None
    except:
        logger.error('Error when loading {}'.format(mp4_path))
        return None, None

    return imgs_transform(imgs, label, mode, seg_num, seglen, \
                 short_size, target_size, img_mean, img_std)


def decode_pickle(sample, mode, seg_num, seglen, short_size, target_size,
                  img_mean, img_std):
    pickle_path = sample[0]
    try:
        if python_ver < (3, 0):
            data_loaded = pickle.load(open(pickle_path, 'rb'))
        else:
            data_loaded = pickle.load(open(pickle_path, 'rb'), encoding='bytes')

        vid, label, frames = data_loaded
        if len(frames) < 1:
            logger.error('{} frame length {} less than 1.'.format(pickle_path,
                                                                  len(frames)))
            return None, None
    except:
        logger.info('Error when loading {}'.format(pickle_path))
        return None, None

    if mode == 'train' or mode == 'valid' or mode == 'test':
        ret_label = label
    elif mode == 'infer':
        ret_label = vid

    imgs = video_loader(frames, seg_num, seglen, mode)
    return imgs_transform(imgs, ret_label, mode, seg_num, seglen, \
                 short_size, target_size, img_mean, img_std)


def imgs_transform(imgs, label, mode, seg_num, seglen, short_size, target_size,
                   img_mean, img_std):
    imgs = group_scale(imgs, short_size)

    if mode == 'train':
        imgs = group_random_crop(imgs, target_size)
        imgs = group_random_flip(imgs)
    else:
        imgs = group_center_crop(imgs, target_size)

    np_imgs = (np.array(imgs[0]).astype('float32').transpose(
        (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
    for i in range(len(imgs) - 1):
        img = (np.array(imgs[i + 1]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
        np_imgs = np.concatenate((np_imgs, img))
    imgs = np_imgs
    imgs -= img_mean
    imgs /= img_std
    imgs = np.reshape(imgs, (seg_num, seglen * 3, target_size, target_size))

    return imgs, label


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
          "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

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
        assert (w >= target_size) and (h >= target_size), \
             "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


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


def imageloader(buf):
    if isinstance(buf, str):
        img = Image.open(StringIO(buf))
    else:
        img = Image.open(BytesIO(buf))

    return img.convert('RGB')


def video_loader(frames, nsample, seglen, mode):
    videolen = len(frames)
    average_dur = int(videolen / nsample)

    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - seglen) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = frames[int(jj % videolen)]
            img = imageloader(imgbuf)
            imgs.append(img)

    return imgs


def mp4_loader(filepath, nsample, seglen, mode):
    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    average_dur = int(videolen / nsample)
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)

    imgs = []
    for i in range(nsample):
        idx = 0
        if mode == 'train':
            if average_dur >= seglen:
                idx = random.randint(0, average_dur - seglen)
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seglen:
                idx = (average_dur - 1) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i

        for jj in range(idx, idx + seglen):
            imgbuf = sampledFrames[int(jj % videolen)]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)

    return imgs
