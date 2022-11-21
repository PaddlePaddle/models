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

from functools import partial
import six
import math
import random
import cv2
import numpy as np
import importlib
from PIL import Image

__all__ = ["Resize", "ResizeByShort", "Normalize", "ToCHWImage", "ExpandDim"]


class ResizeBase(object):
    """
    The base class of resize.
    """
    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, size_divisor=None, interp='LINEAR'):
        if size_divisor is not None:
            assert isinstance(size_divisor,
                              int), "size_divisor should be None or int"
        if interp not in self.interp_dict:
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))

        self.size_divisor = size_divisor
        self.interp = interp

    @staticmethod
    def resize(im, target_size, interp):
        if isinstance(target_size, (list, tuple)):
            w = target_size[0]
            h = target_size[1]
        elif isinstance(target_size, int):
            w = target_size
            h = target_size
        else:
            raise ValueError(
                "target_size should be int (wh, wh), list (w, h) or tuple (w, h)"
            )
        im = cv2.resize(im, (w, h), interpolation=interp)
        return im

    @staticmethod
    def rescale_size(img_size, target_size):
        scale = min(
            max(target_size) / max(img_size), min(target_size) / min(img_size))
        rescaled_size = [round(i * scale) for i in img_size]
        return rescaled_size, scale

    def __call__(self, img):
        raise NotImplementedError


class Resize(ResizeBase):
    """
    Resize an image.

    Args:
        target_size (list|tuple, optional): The target size (w, h) of image. Default: (512, 512).
        keep_ratio (bool, optional): Whether to keep the same ratio for width and height in resizing.
            Default: False.
        size_divisor (int, optional): If size_divisor is not None, make the width and height be the times
            of size_divisor. Default: None.
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".
    """

    def __init__(self,
                 target_size=(512, 512),
                 keep_ratio=False,
                 size_divisor=None,
                 interp='LINEAR'):
        super().__init__(size_divisor=size_divisor, interp=interp)

        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size
        self.keep_ratio = keep_ratio

    def __call__(self, img):
        target_size = self.target_size
        if self.keep_ratio:
            h, w = img.shape[0:2]
            target_size, _ = self.rescale_size((w, h), self.target_size)
        if self.size_divisor:
            target_size = [
                math.ceil(i / self.size_divisor) * self.size_divisor
                for i in target_size
            ]

        img = self.resize(img, target_size, self.interp_dict[self.interp])
        return img


class ResizeByShort(ResizeBase):
    """
    Resize an image by short.

    Args:
        target_size (list|tuple, optional): The target size (w, h) of image. Default: (512, 512).
        size_divisor (int, optional): If size_divisor is not None, make the width and height be the times
            of size_divisor. Default: None.
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']. Note that when it is
            'RANDOM', a random interpolation mode would be specified. Default: "LINEAR".
    """

    def __init__(self, resize_short=512, size_divisor=None, interp='LINEAR'):
        super().__init__(size_divisor=size_divisor, interp=interp)

        self.resize_short = resize_short

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = self.resize_short / min(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        if self.size_divisor is not None:
            h_resize = math.ceil(h_resize /
                                 self.size_divisor) * self.size_divisor
            w_resize = math.ceil(w_resize /
                                 self.size_divisor) * self.size_divisor

        img = self.resize(img, (w_resize, h_resize),
                          self.interp_dict[self.interp])
        return img


class Normalize(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [
            3, 4
        ], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros(
                (1, img_h, img_w)) if self.order == 'chw' else np.zeros(
                    (img_h, img_w, 1))
            img = (np.concatenate(
                (img, pad_zeros), axis=0)
                   if self.order == 'chw' else np.concatenate(
                       (img, pad_zeros), axis=2))
        return img.astype(self.output_dtype)


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))


class ExpandDim(object):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, img):
        img = np.expand_dims(img, axis=self.axis)
        return img
