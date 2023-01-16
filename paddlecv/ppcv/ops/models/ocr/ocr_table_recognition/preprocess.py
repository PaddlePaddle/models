"""
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np


class ResizeTableImage(object):
    def __init__(self, max_len, **kwargs):
        super(ResizeTableImage, self).__init__()
        self.max_len = max_len

    def __call__(self, data):
        img = data['image']
        height, width = img.shape[0:2]
        ratio = self.max_len / (max(height, width) * 1.0)
        resize_h = int(height * ratio)
        resize_w = int(width * ratio)
        resize_img = cv2.resize(img, (resize_w, resize_h))
        data['image'] = resize_img
        data['shape'] = np.array([height, width, ratio, ratio])
        data['max_len'] = self.max_len
        return data


class PaddingTableImage(object):
    def __init__(self, size, **kwargs):
        super(PaddingTableImage, self).__init__()
        self.size = size

    def __call__(self, data):
        img = data['image']
        pad_h, pad_w = self.size
        padding_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        height, width = img.shape[0:2]
        padding_img[0:height, 0:width, :] = img.copy()
        data['image'] = padding_img
        shape = data['shape'].tolist()
        shape.extend([pad_h, pad_w])
        data['shape'] = np.array(shape)
        return data
