# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import cv2
import math
import random
import numpy as np
from PIL import Image, ImageEnhance
import logging
from paddle.io import Dataset

logger = logging.getLogger(__name__)

__all__ = ['KineticsDataset']


class KineticsDataset(Dataset):
    def __init__(self, mode, cfg):
        self.mode = mode
        self.format = cfg.MODEL.format
        self.num_frames = cfg.MODEL.num_frames
        self.sampling_rate = cfg.MODEL.sampling_rate
        self.target_fps = cfg.MODEL.target_fps
        self.slowfast_alpha = cfg.MODEL.alpha

        self.target_size = cfg[mode.upper()]['target_size']
        self.img_mean = cfg.MODEL.image_mean
        self.img_std = cfg.MODEL.image_std
        self.filelist = cfg[mode.upper()]['filelist']

        if self.mode in ["train", "valid"]:
            self.min_size = cfg[mode.upper()]['min_size']
            self.max_size = cfg[mode.upper()]['max_size']
            self.num_ensemble_views = 1
            self.num_spatial_crops = 1
            self._num_clips = 1
        elif self.mode in ['test', 'infer']:
            self.min_size = self.max_size = self.target_size
            self.num_ensemble_views = cfg.TEST.num_ensemble_views
            self.num_spatial_crops = cfg.TEST.num_spatial_crops
            self._num_clips = (self.num_ensemble_views * self.num_spatial_crops)

        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        self._num_retries = 5
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(self.filelist, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                if self.mode == 'infer':
                    path = path_label
                    label = 0  # without label when infer actually
                else:
                    path, label = path_label.split()
                for idx in range(self._num_clips):
                    self._path_to_videos.append(path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)

    def __len__(self):
        return len(self._path_to_videos)

    def __getitem__(self, idx):
        if self.mode in ["train", "valid"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
        elif self.mode in ["test", 'infer']:
            temporal_sample_index = (self._spatial_temporal_idx[idx] //
                                     self.num_spatial_crops)
            spatial_sample_index = (self._spatial_temporal_idx[idx] %
                                    self.num_spatial_crops)

        for ir in range(self._num_retries):
            mp4_path = self._path_to_videos[idx]
            try:
                pathways = self.mp4_loader(
                    mp4_path,
                    temporal_sample_index,
                    spatial_sample_index,
                    temporal_num_clips=self.num_ensemble_views,
                    spatial_num_clips=self.num_spatial_crops,
                    num_frames=self.num_frames,
                    sampling_rate=self.sampling_rate,
                    target_fps=self.target_fps,
                    target_size=self.target_size,
                    img_mean=self.img_mean,
                    img_std=self.img_std,
                    slowfast_alpha=self.slowfast_alpha,
                    min_size=self.min_size,
                    max_size=self.max_size)
            except:
                if ir < self._num_retries - 1:
                    logger.error(
                        'Error when loading {}, have {} trys, will try again'.
                        format(mp4_path, ir))
                    idx = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                else:
                    logger.error(
                        'Error when loading {}, have {} trys, will not try again'.
                        format(mp4_path, ir))
                    return None, None
            label = self._labels[idx]
            return pathways[0], pathways[1], np.array([label]), np.array([idx])

    def mp4_loader(self, filepath, temporal_sample_index, spatial_sample_index,
                   temporal_num_clips, spatial_num_clips, num_frames,
                   sampling_rate, target_fps, target_size, img_mean, img_std,
                   slowfast_alpha, min_size, max_size):
        frames_sample, clip_size = self.decode_sampling(
            filepath, temporal_sample_index, temporal_num_clips, num_frames,
            sampling_rate, target_fps)
        frames_select = self.temporal_sampling(
            frames_sample, clip_size, num_frames, filepath,
            temporal_sample_index, temporal_num_clips)
        frames_resize = self.scale(frames_select, min_size, max_size)
        frames_crop = self.crop(frames_resize, target_size,
                                spatial_sample_index, spatial_num_clips)
        frames_flip = self.flip(frames_crop, spatial_sample_index)

        #list to nparray
        npframes = (np.stack(frames_flip)).astype('float32')
        npframes_norm = self.color_norm(npframes, img_mean, img_std)
        frames_list = self.pack_output(npframes_norm, slowfast_alpha)

        return frames_list

    def get_start_end_idx(self, video_size, clip_size, clip_idx,
                          temporal_num_clips):
        delta = max(video_size - clip_size, 0)
        if clip_idx == -1:  # when test, temporal_num_clips is not used
            # Random temporal sampling.
            start_idx = random.uniform(0, delta)
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / temporal_num_clips
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx

    def decode_sampling(self, filepath, temporal_sample_index,
                        temporal_num_clips, num_frames, sampling_rate,
                        target_fps):
        cap = cv2.VideoCapture(filepath)
        videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)

        clip_size = num_frames * sampling_rate * fps / target_fps

        if filepath[-3:] != 'mp4':
            start_idx, end_idx = 0, math.inf
        else:
            start_idx, end_idx = self.get_start_end_idx(
                videolen, clip_size, temporal_sample_index, temporal_num_clips)
        #print("filepath:",filepath,"start_idx:",start_idx,"end_idx:",end_idx)

        frames_sample = []  #start randomly, decode clip size
        start_idx = math.ceil(start_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for i in range(videolen):
            if i < start_idx:
                continue
            ret, frame = cap.read()
            if ret == False:
                continue
            if i <= end_idx + 1:  #buffer
                img = frame[:, :, ::-1]  #BGR -> RGB
                frames_sample.append(img)
            else:
                break
        return frames_sample, clip_size

    def temporal_sampling(self, frames_sample, clip_size, num_frames, filepath,
                          temporal_sample_index, temporal_num_clips):
        """ sample num_frames from clip_size """
        fs_len = len(frames_sample)

        if filepath[-3:] != 'mp4':
            start_idx, end_idx = self.get_start_end_idx(
                fs_len, clip_size, temporal_sample_index, temporal_num_clips)
        else:
            start_idx, end_idx = self.get_start_end_idx(fs_len, clip_size, 0, 1)

        index = np.linspace(start_idx, end_idx, num_frames).astype("int64")
        index = np.clip(index, 0, fs_len - 1)
        frames_select = []
        for i in range(index.shape[0]):
            idx = index[i]
            imgbuf = frames_sample[idx]
            img = Image.fromarray(imgbuf, mode='RGB')
            frames_select.append(img)

        return frames_select

    def scale(self, frames_select, min_size, max_size):
        size = int(round(np.random.uniform(min_size, max_size)))
        assert (len(frames_select) >= 1) , \
            "len(frames_select):{} should be larger than 1".format(len(frames_select))
        width, height = frames_select[0].size
        if (width <= height and width == size) or (height <= width and
                                                   height == size):
            return frames_select

        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        frames_resize = []
        for j in range(len(frames_select)):
            img = frames_select[j]
            scale_img = img.resize((new_width, new_height), Image.BILINEAR)
            frames_resize.append(scale_img)

        return frames_resize

    def crop(self, frames_resize, target_size, spatial_sample_index,
             spatial_num_clips):
        w, h = frames_resize[0].size
        if w == target_size and h == target_size:
            return frames_resize

        assert (w >= target_size) and (h >= target_size), \
            "image width({}) and height({}) should be larger than crop size({},{})".format(w, h, target_size, target_size)
        frames_crop = []
        if spatial_sample_index == -1:
            x_offset = random.randint(0, w - target_size)
            y_offset = random.randint(0, h - target_size)
        else:
            x_gap = int(math.ceil((w - target_size) / (spatial_num_clips - 1)))
            y_gap = int(math.ceil((h - target_size) / (spatial_num_clips - 1)))
            if h > w:
                x_offset = int(math.ceil((w - target_size) / 2))
                if spatial_sample_index == 0:
                    y_offset = 0
                elif spatial_sample_index == spatial_num_clips - 1:
                    y_offset = h - target_size
                else:
                    y_offset = y_gap * spatial_sample_index
            else:
                y_offset = int(math.ceil((h - target_size) / 2))
                if spatial_sample_index == 0:
                    x_offset = 0
                elif spatial_sample_index == spatial_num_clips - 1:
                    x_offset = w - target_size
                else:
                    x_offset = x_gap * spatial_sample_index

        for img in frames_resize:
            nimg = img.crop((x_offset, y_offset, x_offset + target_size,
                             y_offset + target_size))
            frames_crop.append(nimg)
        return frames_crop

    def flip(self, frames_crop, spatial_sample_index):
        # without flip when test
        if spatial_sample_index != -1:
            return frames_crop

        frames_flip = []
        if np.random.uniform() < 0.5:
            for img in frames_crop:
                nimg = img.transpose(Image.FLIP_LEFT_RIGHT)
                frames_flip.append(nimg)
        else:
            frames_flip = frames_crop
        return frames_flip

    def color_norm(self, npframes_norm, c_mean, c_std):
        npframes_norm /= 255.0
        npframes_norm -= np.array(c_mean).reshape(
            [1, 1, 1, 3]).astype(np.float32)
        npframes_norm /= np.array(c_std).reshape(
            [1, 1, 1, 3]).astype(np.float32)
        return npframes_norm

    def pack_output(self, npframes_norm, slowfast_alpha):
        fast_pathway = npframes_norm

        # sample num points between start and end
        slow_idx_start = 0
        slow_idx_end = fast_pathway.shape[0] - 1
        slow_idx_num = fast_pathway.shape[0] // slowfast_alpha
        slow_idxs_select = np.linspace(slow_idx_start, slow_idx_end,
                                       slow_idx_num).astype("int64")
        slow_pathway = fast_pathway[slow_idxs_select]

        # T H W C -> C T H W.
        slow_pathway = slow_pathway.transpose(3, 0, 1, 2)
        fast_pathway = fast_pathway.transpose(3, 0, 1, 2)

        # slow + fast
        frames_list = [slow_pathway, fast_pathway]
        return frames_list
