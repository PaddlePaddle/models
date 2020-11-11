#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import cv2
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
import paddle.fluid as fluid

from PIL import Image, ImageEnhance
import logging

from .reader_utils import DataReader

logger = logging.getLogger(__name__)
python_ver = sys.version_info


class VideoRecord(object):
    '''
    define a class method which used to describe the frames information of videos
    1. self._data[0] is the frames' path
    2. self._data[1] is the number of frames 
    3. self._data[2] is the label of frames
    '''

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


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
        super(KineticsReader, self).__init__(name, mode, cfg)
        self.format = cfg.MODEL.format
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.seg_num = self.get_config_from_sec('model', 'seg_num')
        self.seglen = self.get_config_from_sec('model', 'seglen')

        self.seg_num = self.get_config_from_sec(mode, 'seg_num', self.seg_num)
        self.short_size = self.get_config_from_sec(mode, 'short_size')
        self.target_size = self.get_config_from_sec(mode, 'target_size')
        self.num_reader_threads = self.get_config_from_sec(mode,
                                                           'num_reader_threads')
        self.buf_size = self.get_config_from_sec(mode, 'buf_size')
        self.fix_random_seed = self.get_config_from_sec(mode, 'fix_random_seed')

        self.img_mean = np.array(cfg.MODEL.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape(
            [3, 1, 1]).astype(np.float32)
        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.filelist = cfg[mode.upper()]['filelist']

        if self.fix_random_seed:
            random.seed(0)
            np.random.seed(0)
            self.num_reader_threads = 1

    def create_reader(self):
        assert os.path.exists(self.filelist), \
                    '{} not exist, please check the data list'.format(self.filelist)
        _reader = self._reader_creator(self.filelist, self.mode, seg_num=self.seg_num, seglen = self.seglen, \
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

    def _reader_creator(self,
                        file_list,
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
                        format='frames'):
        def decode_mp4(sample, mode, seg_num, seglen, short_size, target_size,
                       img_mean, img_std):
            sample = sample[0].split(' ')
            mp4_path = sample[0]
            if mode == "infer":
                label = mp4_path.split('/')[-1]
            else:
                label = int(sample[1])
            try:
                imgs = mp4_loader(mp4_path, seg_num, seglen, mode)
                if len(imgs) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        mp4_path, len(imgs)))
                    return None, None
            except:
                logger.error('Error when loading {}'.format(mp4_path))
                return None, None

            return imgs_transform(imgs, mode, seg_num, seglen, \
                         short_size, target_size, img_mean, img_std, name = self.name), label

        def decode_frames(sample, mode, seg_num, seglen, short_size,
                          target_size, img_mean, img_std):
            recode = VideoRecord(sample[0].split(' '))
            frames_dir_path = recode.path
            if mode == "infer":
                label = frames_dir_path
            else:
                label = recode.label

            try:
                imgs = frames_loader(recode, seg_num, seglen, mode)
                if len(imgs) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        frames_dir_path, len(imgs)))
                    return None, None
            except:
                logger.error('Error when loading {}'.format(frames_dir_path))
                return None, None

            return imgs_transform(
                imgs,
                mode,
                seg_num,
                seglen,
                short_size,
                target_size,
                img_mean,
                img_std,
                name=self.name), label

        def reader_():
            with open(file_list) as flist:
                lines = [line.strip() for line in flist]
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    file_path = line.strip()
                    yield [file_path]

        if format == 'frames':
            decode_func = decode_frames
        elif format == 'video':
            decode_func = decode_mp4
        else:
            raise ("Not implemented format {}".format(format))

        mapper = functools.partial(
            decode_func,
            mode=mode,
            seg_num=seg_num,
            seglen=seglen,
            short_size=short_size,
            target_size=target_size,
            img_mean=img_mean,
            img_std=img_std)

        return fluid.io.xmap_readers(
            mapper, reader_, num_threads, buf_size, order=True)


def imgs_transform(imgs,
                   mode,
                   seg_num,
                   seglen,
                   short_size,
                   target_size,
                   img_mean,
                   img_std,
                   name=''):
    imgs = group_scale(imgs, short_size)

    np_imgs = np.array([np.array(img).astype('float32') for img in imgs])  #dhwc

    if mode == 'train':
        np_imgs = group_crop(np_imgs, target_size)
        np_imgs = group_random_flip(np_imgs)
    else:
        np_imgs = group_crop(np_imgs, target_size, is_center=True)

    np_imgs = np_imgs.transpose(0, 3, 1, 2) / 255  #dchw
    np_imgs -= img_mean
    np_imgs /= img_std

    return np_imgs


def group_crop(np_imgs, target_size, is_center=True):
    d, h, w, c = np_imgs.shape
    th, tw = target_size, target_size
    assert (w >= target_size) and (h >= target_size), \
          "image width({}) and height({}) should be larger than crop size".format(w, h, target_size)

    if is_center:
        h_off = int(round((h - th) / 2.))
        w_off = int(round((w - tw) / 2.))
    else:
        w_off = random.randint(0, w - tw)
        h_off = random.randint(0, h - th)

    img_crop = np_imgs[:, h_off:h_off + target_size, w_off:w_off +
                       target_size, :]
    return img_crop


def group_random_flip(np_imgs):
    prob = random.random()
    if prob < 0.5:
        ret = np_imgs[:, :, ::-1, :]
        return ret
    else:
        return np_imgs


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


def mp4_loader(filepath, nsample, seglen, mode):
    cap = cv2.VideoCapture(filepath)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)
    average_dur = int(len(sampledFrames) / nsample)
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
            imgbuf = sampledFrames[int(jj % len(sampledFrames))]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)

    return imgs


def frames_loader(recode, nsample, seglen, mode):
    imgpath, num_frames = recode.path, recode.num_frames
    average_dur = int(num_frames / nsample)
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
            img = Image.open(
                os.path.join(imgpath, 'img_{:05d}.jpg'.format(jj + 1))).convert(
                    'RGB')
            imgs.append(img)
    return imgs
