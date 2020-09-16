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
from PIL import Image


class Scale(object):
    """
    Scale images.

    Args:
        short_size(float | int): Short size of an image will be scaled to the short_size.
    """

    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, imgs):
        """
        Performs resize operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        resized_imgs = []
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.size
            if (w <= h and w == self.short_size) or (h <= w and
                                                     h == self.short_size):
                resized_imgs.append(img)
                continue

            if w < h:
                ow = self.short_size
                oh = int(self.short_size * 4.0 / 3.0)
                resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
            else:
                oh = self.short_size
                ow = int(self.short_size * 4.0 / 3.0)
                resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

        return resized_imgs


class RandomCrop(object):
    """
    Random crop images.

    Args:
        target_size(int): Random crop a square with the target_size from an image.
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, imgs):
        """
        Performs random crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after random crop.
        """
        w, h = imgs[0].size
        th, tw = self.target_size, self.target_size

        assert (w >= self.target_size) and (h >= self.target_size), \
            "image width({}) and height({}) should be larger than crop size".format(
                w, h, self.target_size)

        crop_images = []
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in imgs:
            if w == tw and h == th:
                crop_images.append(img)
            else:
                crop_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return crop_images


class CenterCrop(object):
    """
    Center crop images.

    Args:
        target_size(int): Center crop a square with the target_size from an image.
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, imgs):
        """
        Performs Center crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            ccrop_imgs: List where each item is a PIL.Image after Center crop.
        """
        ccrop_imgs = []
        for img in imgs:
            w, h = img.size
            th, tw = self.target_size, self.target_size
            assert (w >= self.target_size) and (h >= self.target_size), \
                "image width({}) and height({}) should be larger than crop size".format(
                    w, h, self.target_size)
            x1 = int(round((w - tw) / 2.))
            y1 = int(round((h - th) / 2.))
            ccrop_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return ccrop_imgs


class RandomFlip(object):
    """
    Random Flip images.

    Args:
        p(float): Random flip images with the probability p.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Performs random flip operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            flip_imgs: List where each item is a PIL.Image after random flip.
        """
        v = random.random()
        if v < self.p:
            flip_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
            return flip_imgs
        else:
            return imgs


class Image2Array(object):
    """
    transfer PIL.Image to Numpy array and transpose dimensions from 'dhwc' to 'dchw'.
    """

    def __init__(self):
        self.format = "dhwc"

    def __call__(self, imgs):
        """
        Performs Image to NumpyArray operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            np_imgs: Numpy array.
        """
        np_imgs = np.array(
            [np.array(img).astype('float32') for img in imgs])  #dhwc
        np_imgs = np_imgs.transpose(0, 3, 1, 2)  #dchw
        return np_imgs


class Normalization(object):
    """
    Normalization.
    Args:
        mean(list[float]): mean values of different channels.
        std(list[float]): std values of differetn channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        """
        Performs normalization operations.
        Args:
            imgs: Numpy array.
        return:
            np_imgs: Numpy array after normalization.
        """
        norm_imgs = imgs / 255
        norm_imgs -= self.mean
        norm_imgs /= self.std
        return norm_imgs
