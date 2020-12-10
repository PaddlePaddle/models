#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import random
import numpy as np
import logging
from paddle.io import Dataset
from augmentations import *
from loader import *

logger = logging.getLogger(__name__)


class TSN_UCF101_Dataset(Dataset):
    def __init__(self, cfg, mode):
        self.mode = mode
        self.format = cfg.MODEL.format  #'videos' or 'frames'
        self.seg_num = cfg.MODEL.seg_num
        self.seglen = cfg.MODEL.seglen
        self.short_size = cfg.TRAIN.short_size
        self.target_size = cfg.TRAIN.target_size
        self.img_mean = np.array(cfg.MODEL.image_mean).reshape(
            [3, 1, 1]).astype(np.float32)
        self.img_std = np.array(cfg.MODEL.image_std).reshape(
            [3, 1, 1]).astype(np.float32)

        self.filelist = cfg[mode.upper()]['filelist']

        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        self._num_retries = 5
        self._path_to_videos = []
        self._labels = []
        self._num_frames = []
        with open(self.filelist, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                if self.format == "videos":
                    path, label = path_label.split()
                    self._path_to_videos.append(path + '.avi')
                    self._num_frames.append(0)  # unused 
                    self._labels.append(int(label))
                elif self.format == "frames":
                    path, num_frames, label = path_label.split()
                    self._path_to_videos.append(path)
                    self._num_frames.append(int(num_frames))
                    self._labels.append(int(label))

    def __len__(self):
        return len(self._path_to_videos)

    def __getitem__(self, idx):
        for ir in range(self._num_retries):
            path = self._path_to_videos[idx]
            num_frames = self._num_frames[idx]
            try:
                frames = self.pipline(
                    path,
                    num_frames,
                    format=self.format,
                    seg_num=self.seg_num,
                    seglen=self.seglen,
                    short_size=self.short_size,
                    target_size=self.target_size,
                    img_mean=self.img_mean,
                    img_std=self.img_std,
                    mode=self.mode)
            except:
                if ir < self._num_retries - 1:
                    logger.error(
                        'Error when loading {}, have {} trys, will try again'.
                        format(path, ir))
                    idx = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                else:
                    logger.error(
                        'Error when loading {}, have {} trys, will not try again'.
                        format(path, ir))
                    return None, None
            label = self._labels[idx]
            return frames, np.array([label])  #, np.array([idx])

    def pipline(self, filepath, num_frames, format, seg_num, seglen, short_size,
                target_size, img_mean, img_std, mode):
        #Loader
        if format == 'videos':
            Loader_ops = [
                VideoDecoder(filepath), VideoSampler(seg_num, seglen, mode)
            ]
        elif format == 'frames':
            Loader_ops = [
                FrameLoader(filepath, num_frames, seg_num, seglen, mode)
            ]

        #Augmentation
        if mode == 'train':
            Aug_ops = [
                Scale(short_size), RandomCrop(target_size), RandomFlip(),
                Image2Array(), Normalization(img_mean, img_std)
            ]
        else:
            Aug_ops = [
                Scale(short_size), CenterCrop(target_size), Image2Array(),
                Normalization(img_mean, img_std)
            ]

        ops = Loader_ops + Aug_ops
        data = ops[0]()
        for op in ops[1:]:
            data = op(data)
        return data
