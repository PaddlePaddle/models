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

import os
import cv2
import random
from PIL import Image


class VideoDecoder(object):
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __call__(self):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        cap = cv2.VideoCapture(self.filepath)
        videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampledFrames = []
        for i in range(videolen):
            ret, frame = cap.read()
            # maybe first frame is empty
            if ret == False:
                continue
            img = frame[:, :, ::-1]
            sampledFrames.append(img)
        return sampledFrames


class VideoSampler(object):
    """
    Sample frames.
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        mode(str): 'train', 'test' or 'infer'

    """

    def __init__(self, num_seg, seg_len, mode):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.mode = mode

    def __call__(self, frames):
        """
        Args:
            frames: List where each item is a numpy array decoding from video.
        return:
            List where each item is a PIL.Image after sampling.
        """
        average_dur = int(len(frames) / self.num_seg)
        imgs = []
        for i in range(self.num_seg):
            idx = 0
            if self.mode == 'train':
                if average_dur >= self.seg_len:
                    idx = random.randint(0, average_dur - self.seg_len)
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            else:
                if average_dur >= self.seg_len:
                    idx = (average_dur - 1) // 2
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i

            for jj in range(idx, idx + self.seg_len):
                imgbuf = frames[int(jj % len(frames))]
                img = Image.fromarray(imgbuf, mode='RGB')
                imgs.append(img)
        return imgs


class FrameLoader(object):
    """
    Load frames.
    Args:
        filepath(str): the file path of frames file.
        num_frames(int): number of frames in a video file.
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        mode(str): 'train', 'test' or 'infer'.
    """

    def __init__(self, filepath, num_frames, num_seg, seg_len, mode):
        self.filepath = filepath
        self.num_frames = num_frames
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.mode = mode

    def __call__(self):
        """
        return:
            imgs: List where each item is a PIL.Image.
        """
        average_dur = int(self.num_frames / self.num_seg)
        imgs = []
        for i in range(self.num_seg):
            idx = 0
            if self.mode == 'train':
                if average_dur >= self.seg_len:
                    idx = random.randint(0, average_dur - self.seg_len)
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            else:
                if average_dur >= self.seg_len:
                    idx = (average_dur - 1) // 2
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i

            for jj in range(idx, idx + self.seg_len):
                img = Image.open(
                    os.path.join(self.filepath, 'img_{:05d}.jpg'.format(
                        jj + 1))).convert('RGB')
                imgs.append(img)
        return imgs
