# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

# code reference: https://github.com/huangjun12/KeyFramesExtraction/blob/master/scene_div.py

import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import argrelextrema


def smooth(x, window_len=13, window='hanning'):
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:
                                                              -1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


class Frame:
    """class to hold information about each frame
    """

    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
    x = (b - a) / max(a, b)
    return x


class KeyFrameExtractor(object):
    def __init__(self, config):
        pass

    def extract_by_video_path(self, video_path):
        raise NotImplementedError

    def exrtact_by_image_list(self, image_list):
        raise NotImplementedError

    def __call__(self, video_obj):
        assert isinstance(video_obj, (list, tuple, str))
        if isinstance(video_obj, str):
            output = self.extract_by_video_path(video_obj)
        elif isinstance(video_obj, (list, tuple)):
            output = self.exrtact_by_image_list(video_obj)
        return output


class LUVAbsDiffKeyFrameExtractor(KeyFrameExtractor):
    """
    extract key frames based on sum of absolute differences in LUV colorspace.
    """

    def __init__(self, config):
        self.thresh = config.get("thresh", None)
        self.use_top_order = config.get("use_top_order", False)
        self.use_local_maxima = config.get("use_local_maxima", None)
        self.num_top_frames = config.get("num_top_frames", None)
        self.window_len = config.get("window_len", None)

    def extract_by_video_path(self, video_path):
        cap = cv2.VideoCapture(video_path)
        curr_frame = None
        prev_frame = None
        frame_diffs = []
        frames = []
        success, frame = cap.read()
        i = 0
        while (success):
            luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            curr_frame = luv
            if curr_frame is not None and prev_frame is not None:
                diff = cv2.absdiff(curr_frame, prev_frame)
                diff_sum = np.sum(diff)
                diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
                frame_diffs.append(diff_sum_mean)
                frame = Frame(i, diff_sum_mean)
                frames.append(frame)
            prev_frame = curr_frame
            i = i + 1
            success, frame = cap.read()
        cap.release()

        # compute keyframe
        keyframe_id_set = set()
        if self.use_top_order:
            # sort the list in descending order
            frames.sort(key=operator.attrgetter("diff"), reverse=True)
            for keyframe in frames[:self.num_top_frames]:
                keyframe_id_set.add(keyframe.id)
        if self.thresh is not None:
            for i in range(1, len(frames)):
                if (rel_change(
                        np.float(frames[i - 1].diff), np.float(frames[i].diff))
                        >= self.thresh):
                    keyframe_id_set.add(frames[i].id)
        if self.use_local_maxima:
            diff_array = np.array(frame_diffs)
            sm_diff_array = smooth(diff_array, self.window_len)
            frame_indexes = np.asarray(
                argrelextrema(sm_diff_array, np.greater))[0]
            for i in frame_indexes:
                keyframe_id_set.add(frames[i - 1].id)

        keyframe_id_set = sorted(list(keyframe_id_set))
        # save all keyframes as image
        cap = cv2.VideoCapture(str(video_path))
        curr_frame = None
        keyframes = []
        success, frame = cap.read()
        idx = 0
        while (success):
            if idx in keyframe_id_set:
                keyframes.append(frame)
            idx = idx + 1
            success, frame = cap.read()
        cap.release()
        return keyframes, keyframe_id_set
