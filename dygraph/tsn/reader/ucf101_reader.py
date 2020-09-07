# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import cv2
import math
import random
import functools
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
        return "./data/dataset/ucf101/rawframes/" + self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class UCF101Reader(DataReader):
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
        super(UCF101Reader, self).__init__(name, mode, cfg)
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
        # set num_trainers and trainer_id when distributed training is implemented
        self.num_trainers = self.get_config_from_sec(mode, 'num_trainers', 1)
        self.trainer_id = self.get_config_from_sec(mode, 'trainer_id', 0)

        if self.mode == 'infer':
            self.video_path = cfg[mode.upper()]['video_path']
        else:
            self.video_path = ''
        if self.fix_random_seed:
            random.seed(0)
            np.random.seed(0)
            self.num_reader_threads = 1

    def create_reader(self):

        # if set video_path for inference mode, just load this single video
        if (self.mode == 'infer') and (self.video_path != ''):
            # load video from file stored at video_path
            _reader = self._inference_reader_creator(
                self.video_path,
                self.mode,
                seg_num=self.seg_num,
                seglen=self.seglen,
                short_size=self.short_size,
                target_size=self.target_size,
                img_mean=self.img_mean,
                img_std=self.img_std)
        else:
            assert os.path.exists(self.filelist), \
                '{} not exist, please check the data list'.format(
                    self.filelist)
            _reader = self._reader_creator(
                self.filelist,
                self.mode,
                seg_num=self.seg_num,
                seglen=self.seglen,
                short_size=self.short_size,
                target_size=self.target_size,
                img_mean=self.img_mean,
                img_std=self.img_std,
                shuffle=(self.mode == 'train'),
                num_threads=self.num_reader_threads,
                buf_size=self.buf_size,
                format=self.format)

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

    def _inference_reader_creator(self, video_path, mode, seg_num, seglen,
                                  short_size, target_size, img_mean, img_std):
        def reader():
            try:
                imgs = mp4_loader(video_path, seg_num, seglen, mode)
                if len(imgs) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        video_path, len(imgs)))
                    yield None, None
            except:
                logger.error('Error when loading {}'.format(mp4_path))
                yield None, None

            imgs_ret = imgs_transform(imgs, mode, seg_num, seglen, short_size,
                                      target_size, img_mean, img_std)
            label_ret = video_path

            yield imgs_ret, label_ret

        return reader

    def _reader_creator(self,
                        pickle_list,
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
                        format='avi'):
        def decode_mp4(sample, mode, seg_num, seglen, short_size, target_size,
                       img_mean, img_std):
            sample = sample[0].split(' ')
            mp4_path = "./data/dataset/ucf101/videos/" + sample[0] + ".avi"

            # when infer, we store vid as label
            label = int(sample[1]) - 1
            try:
                imgs = mp4_loader(mp4_path, seg_num, seglen, mode)
                if len(imgs) < 1:
                    logger.error('{} frame length {} less than 1.'.format(
                        mp4_path, len(imgs)))
                    return None, None
            except:
                logger.error('Error when loading {}'.format(mp4_path))
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

        def decode_frames(sample, mode, seg_num, seglen, short_size,
                          target_size, img_mean, img_std):
            recode = VideoRecord(sample[0].split(' '))
            frames_dir_path = recode.path
            # when infer, we store vid as label
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
            with open(pickle_list) as flist:
                full_lines = [line.strip() for line in flist]
                if self.mode == 'train':
                    if (not hasattr(reader_, 'seed')):
                        reader_.seed = 0
                    random.Random(reader_.seed).shuffle(full_lines)
                    print("reader shuffle seed", reader_.seed)
                    if reader_.seed is not None:
                        reader_.seed += 1

                per_node_lines = int(
                    math.ceil(len(full_lines) * 1.0 / self.num_trainers))
                total_lines = per_node_lines * self.num_trainers

                # aligned full_lines so that it can evenly divisible
                full_lines += full_lines[:(total_lines - len(full_lines))]
                assert len(full_lines) == total_lines

                # trainer get own sample
                lines = full_lines[self.trainer_id:total_lines:
                                   self.num_trainers]
                logger.info("trainerid %d, trainer_count %d" %
                            (self.trainer_id, self.num_trainers))
                logger.info(
                    "read images from %d, length: %d, lines length: %d, total: %d"
                    % (self.trainer_id * per_node_lines, per_node_lines,
                       len(lines), len(full_lines)))
                assert len(lines) == per_node_lines
                for line in lines:
                    pickle_path = line.strip()
                    yield [pickle_path]

        if format == 'frames':
            decode_func = decode_frames
        elif format == 'videos':
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

        return fluid.io.xmap_readers(mapper, reader_, num_threads, buf_size)


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

    return imgs


def group_multi_scale_crop(img_group,
                           target_size,
                           scales=None,
                           max_distort=1,
                           fix_crop=True,
                           more_fix_crop=True):
    scales = scales if scales is not None else [1, .875, .75, .66]
    input_size = [target_size, target_size]

    im_size = img_group[0].size

    # get random crop offset
    def _sample_crop_size(im_size):
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in scales]
        crop_h = [
            input_size[1] if abs(x - input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            input_size[0] if abs(x - input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_step = (image_w - crop_pair[0]) / 4
            h_step = (image_h - crop_pair[1]) / 4

            ret = list()
            ret.append((0, 0))  # upper left
            if w_step != 0:
                ret.append((4 * w_step, 0))  # upper right
            if h_step != 0:
                ret.append((0, 4 * h_step))  # lower left
            if h_step != 0 and w_step != 0:
                ret.append((4 * w_step, 4 * h_step))  # lower right
            if h_step != 0 or w_step != 0:
                ret.append((2 * w_step, 2 * h_step))  # center

            if more_fix_crop:
                ret.append((0, 2 * h_step))  # center left
                ret.append((4 * w_step, 2 * h_step))  # center right
                ret.append((2 * w_step, 4 * h_step))  # lower center
                ret.append((2 * w_step, 0 * h_step))  # upper center

                ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

            w_offset, h_offset = random.choice(ret)

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
    crop_img_group = [
        img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        for img in img_group
    ]
    ret_img_group = [
        img.resize((input_size[0], input_size[1]), Image.BILINEAR)
        for img in crop_img_group
    ]

    return ret_img_group


def group_random_crop(img_group, target_size):
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
        "image width({}) and height({}) should be larger than crop size".format(
            w, h, target_size)

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
            "image width({}) and height({}) should be larger than crop size".format(
                w, h, target_size)
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


def load_image(directory, idx):
    return Image.open(os.path.join(directory, 'img_{:05d}.jpg'.format(
        idx))).convert('RGB')


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
            img = load_image(imgpath, jj + 1)
            imgs.append(img)
    return imgs
