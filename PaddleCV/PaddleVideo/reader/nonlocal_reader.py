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
import random
import time
import multiprocessing
import numpy as np
import cv2
import logging

from .reader_utils import DataReader

logger = logging.getLogger(__name__)


class NonlocalReader(DataReader):
    """
    Data reader for kinetics dataset, which read mp4 file and decode into numpy.
    This is for nonlocal neural network model.
          cfg:  num_classes
                num_reader_threads
                image_mean
                image_std
                batch_size
                filelist
                crop_size
                sample_rate
                video_length
                jitter_scales
                Test only cfg: num_test_clips
                               use_multi_crop
    """

    def __init__(self, name, mode, cfg):
        self.name = name
        self.mode = mode
        self.cfg = cfg

    def create_reader(self):
        cfg = self.cfg
        mode = self.mode
        num_reader_threads = cfg[mode.upper()]['num_reader_threads']
        assert num_reader_threads >=1, \
                "number of reader threads({}) should be a positive integer".format(num_reader_threads)
        if num_reader_threads == 1:
            reader_func = make_reader
        else:
            reader_func = make_multi_reader

        dataset_args = {}
        dataset_args['image_mean'] = cfg.MODEL.image_mean
        dataset_args['image_std'] = cfg.MODEL.image_std
        dataset_args['crop_size'] = cfg[mode.upper()]['crop_size']
        dataset_args['sample_rate'] = cfg[mode.upper()]['sample_rate']
        dataset_args['video_length'] = cfg[mode.upper()]['video_length']
        dataset_args['min_size'] = cfg[mode.upper()]['jitter_scales'][0]
        dataset_args['max_size'] = cfg[mode.upper()]['jitter_scales'][1]
        dataset_args['num_reader_threads'] = num_reader_threads
        filelist = cfg[mode.upper()]['filelist']
        batch_size = cfg[mode.upper()]['batch_size']

        if (self.mode == 'infer') and (cfg['INFER']['video_path'] != ''):
            filelist = create_tmp_inference_file(cfg['INFER']['video_path'])

        if self.mode == 'train':
            sample_times = 1
            return reader_func(filelist, batch_size, sample_times, True, True,
                               **dataset_args)
        elif self.mode == 'valid':
            sample_times = 1
            return reader_func(filelist, batch_size, sample_times, False, False,
                               **dataset_args)
        elif self.mode == 'test' or self.mode == 'infer':
            sample_times = cfg['TEST']['num_test_clips']
            if cfg['TEST']['use_multi_crop'] == 1:
                sample_times = int(sample_times / 3)
            if cfg['TEST']['use_multi_crop'] == 2:
                sample_times = int(sample_times / 6)
            return reader_func(filelist, batch_size, sample_times, False, False,
                               **dataset_args)
        else:
            logger.info('Not implemented')
            raise NotImplementedError


def create_tmp_inference_file(video_path,
                              file_path='temp_nonlocal_inference_list'):
    tmp_file = open(file_path, 'w')
    for i in range(10):
        for j in range(3):
            tmp_file.write('{} {} {} {}\n'.format(video_path, 0, i, j))
    tmp_file.close()
    return file_path


def video_fast_get_frame(video_path,
                         sampling_rate=1,
                         length=64,
                         start_frm=-1,
                         sample_times=1):
    cap = cv2.VideoCapture(video_path)
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sampledFrames = []

    video_output = np.ndarray(shape=[length, height, width, 3], dtype=np.uint8)

    use_start_frm = start_frm
    if start_frm < 0:
        if (frame_cnt - length * sampling_rate > 0):
            use_start_frm = random.randint(0,
                                           frame_cnt - length * sampling_rate)
        else:
            use_start_frm = 0
    else:
        frame_gaps = float(frame_cnt) / float(sample_times)
        use_start_frm = int(frame_gaps * start_frm) % frame_cnt

    for i in range(frame_cnt):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)

    for idx in range(length):
        i = use_start_frm + idx * sampling_rate
        i = i % len(sampledFrames)
        video_output[idx] = sampledFrames[i]

    cap.release()
    return video_output


def apply_resize(rgbdata, min_size, max_size):
    length, height, width, channel = rgbdata.shape
    ratio = 1.0
    # generate random scale between [min_size, max_size]
    if min_size == max_size:
        side_length = min_size
    else:
        side_length = np.random.randint(min_size, max_size)
    if height > width:
        ratio = float(side_length) / float(width)
    else:
        ratio = float(side_length) / float(height)
    out_height = int(round(height * ratio))
    out_width = int(round(width * ratio))
    outdata = np.zeros(
        (length, out_height, out_width, channel), dtype=rgbdata.dtype)
    for i in range(length):
        outdata[i] = cv2.resize(rgbdata[i], (out_width, out_height))
    return outdata


def crop_mirror_transform(rgbdata,
                          mean,
                          std,
                          cropsize=224,
                          use_mirror=True,
                          center_crop=False,
                          spatial_pos=-1):
    channel, length, height, width = rgbdata.shape
    assert height >= cropsize, "crop size should not be larger than video height"
    assert width >= cropsize, "crop size should not be larger than video width"
    # crop to specific scale
    if center_crop:
        h_off = int((height - cropsize) / 2)
        w_off = int((width - cropsize) / 2)
        if spatial_pos >= 0:
            now_pos = spatial_pos % 3
            if h_off > 0:
                h_off = h_off * now_pos
            else:
                w_off = w_off * now_pos
    else:
        h_off = np.random.randint(0, height - cropsize)
        w_off = np.random.randint(0, width - cropsize)
    outdata = np.zeros(
        (channel, length, cropsize, cropsize), dtype=rgbdata.dtype)
    outdata[:, :, :, :] = rgbdata[:, :, h_off:h_off + cropsize, w_off:w_off +
                                  cropsize]
    # apply mirror
    mirror_indicator = (np.random.rand() > 0.5)
    mirror_me = use_mirror and mirror_indicator
    if spatial_pos > 0:
        mirror_me = (int(spatial_pos / 3) > 0)
    if mirror_me:
        outdata = outdata[:, :, :, ::-1]
    # substract mean and divide std
    outdata = outdata.astype(np.float32)
    outdata = (outdata - mean) / std
    return outdata


def make_reader(filelist, batch_size, sample_times, is_training, shuffle,
                **dataset_args):
    def reader():
        fl = open(filelist).readlines()
        fl = [line.strip() for line in fl if line.strip() != '']

        if shuffle:
            random.shuffle(fl)

        batch_out = []
        for line in fl:
            # start_time = time.time()
            line_items = line.split(' ')
            fn = line_items[0]
            label = int(line_items[1])
            if len(line_items) > 2:
                start_frm = int(line_items[2])
                spatial_pos = int(line_items[3])
                in_sample_times = sample_times
            else:
                start_frm = -1
                spatial_pos = -1
                in_sample_times = 1
            label = np.array([label]).astype(np.int64)
            # 1, get rgb data for fixed length of frames
            try:
                rgbdata = video_fast_get_frame(fn, \
                             sampling_rate = dataset_args['sample_rate'], length = dataset_args['video_length'], \
                             start_frm = start_frm, sample_times = in_sample_times)
            except:
                logger.info('Error when loading {}, just skip this file'.format(
                    fn))
                continue
            # add prepocessing
            # 2, reszie to randomly scale between [min_size, max_size] when training, or cgf.TEST.SCALE when inference
            min_size = dataset_args['min_size']
            max_size = dataset_args['max_size']
            rgbdata = apply_resize(rgbdata, min_size, max_size)
            # transform [length, height, width, channel] to [channel, length, height, width]
            rgbdata = np.transpose(rgbdata, [3, 0, 1, 2])

            # 3 crop, mirror and transform
            rgbdata = crop_mirror_transform(rgbdata, mean = dataset_args['image_mean'], \
                             std = dataset_args['image_std'], cropsize = dataset_args['crop_size'], \
                             use_mirror = is_training, center_crop = (not is_training), \
                             spatial_pos = spatial_pos)

            batch_out.append((rgbdata, label))
            if len(batch_out) == batch_size:
                yield batch_out
                batch_out = []

    return reader


def make_multi_reader(filelist, batch_size, sample_times, is_training, shuffle,
                      **dataset_args):
    def read_into_queue(flq, queue):
        batch_out = []
        for line in flq:
            line_items = line.split(' ')
            fn = line_items[0]
            label = int(line_items[1])
            if len(line_items) > 2:
                start_frm = int(line_items[2])
                spatial_pos = int(line_items[3])
                in_sample_times = sample_times
            else:
                start_frm = -1
                spatial_pos = -1
                in_sample_times = 1
            label = np.array([label]).astype(np.int64)
            # 1, get rgb data for fixed length of frames
            try:
                rgbdata = video_fast_get_frame(fn, \
                             sampling_rate = dataset_args['sample_rate'], length = dataset_args['video_length'], \
                             start_frm = start_frm, sample_times = in_sample_times)
            except:
                logger.info('Error when loading {}, just skip this file'.format(
                    fn))
                continue
            # add prepocessing
            # 2, reszie to randomly scale between [min_size, max_size] when training, or cgf.TEST.SCALE when inference
            min_size = dataset_args['min_size']
            max_size = dataset_args['max_size']
            rgbdata = apply_resize(rgbdata, min_size, max_size)
            # transform [length, height, width, channel] to [channel, length, height, width]
            rgbdata = np.transpose(rgbdata, [3, 0, 1, 2])

            # 3 crop, mirror and transform
            rgbdata = crop_mirror_transform(rgbdata, mean = dataset_args['image_mean'], \
                             std = dataset_args['image_std'], cropsize = dataset_args['crop_size'], \
                             use_mirror = is_training, center_crop = (not is_training), \
                             spatial_pos = spatial_pos)

            batch_out.append((rgbdata, label))
            if len(batch_out) == batch_size:
                queue.put(batch_out)
                batch_out = []
        queue.put(None)

    def queue_reader():
        # split file list and shuffle
        fl = open(filelist).readlines()
        fl = [line.strip() for line in fl if line.strip() != '']

        if shuffle:
            random.shuffle(fl)

        n = dataset_args['num_reader_threads']
        queue_size = 20
        reader_lists = [None] * n
        file_num = int(len(fl) // n)
        for i in range(n):
            if i < len(reader_lists) - 1:
                tmp_list = fl[i * file_num:(i + 1) * file_num]
            else:
                tmp_list = fl[i * file_num:]
            reader_lists[i] = tmp_list

        queue = multiprocessing.Queue(queue_size)
        p_list = [None] * len(reader_lists)
        # for reader_list in reader_lists:
        for i in range(len(reader_lists)):
            reader_list = reader_lists[i]
            p_list[i] = multiprocessing.Process(
                target=read_into_queue, args=(reader_list, queue))
            p_list[i].start()
        reader_num = len(reader_lists)
        finish_num = 0
        while finish_num < reader_num:
            sample = queue.get()
            if sample is None:
                finish_num += 1
            else:
                yield sample
        for i in range(len(p_list)):
            if p_list[i].is_alive():
                p_list[i].join()

    return queue_reader
