#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
from PIL import Image
import numpy as np
import warnings

__all__ = [
    "Compose", "Resize", "Scale", "RandomHorizontalFlip", "RandomResizedCrop",
    "CenterCrop"
]


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def crop(img, i, j, h, w):
    if not _is_pil_image(img):
        raise TypeError('img should be a PIL Image, but be {}'.format(
            type(img)))
    return img.crop((j, i, j + w, i + h))


def resize(img, size, interpolation=Image.BILINEAR):
    if not _is_pil_image(img):
        raise TypeError('img should be a PIL Image, but be {}'.format(
            type(img)))
    if not (isinstance(size, int) or
            (isinstance(size, tuple) and len(size) == 2)):
        raise TypeError('Wrong size arg: {}'.format(size))
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def center_crop(img, output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


class Compose(object):
    """Make some transforms in a chain.

    Args:
        transforms (list): list of transforms to be in a chain.
    """

    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, img):
        for t in self._transforms:
            img = t(img)
        return img


class Resize(object):
    """Resize the input PIL Image.

    Args:
        size (tuple | int): Output size. If the size is a tuple,
            resize image to that size (h, w). If the size is an int,
            smaller edge of the image will be resized to this size.
        interpolation (int): Interpolation method.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, tuple) and
                                         len(size) == 2)
        self._size = size
        self._interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be resized.

        Returns:
            PIL Image: Resized image.
        """
        return resize(img, self._size, self._interpolation)


class Scale(Resize):
    """
    Note: This transform is deprecated in favor of Resize.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The use of the transforms.Scale transform is deprecated, " +
            "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            if not _is_pil_image(img):
                raise TypeError('img should be a PIL image, but be {}'.format(
                    type(img)))
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomResizedCrop(object):
    """Crop the input PIL Image to random size and aspect ratio, and then
       resize the PIL Image to target size.

    Args:
        size: target size
        scale: range of ratio of the origin size to be cropped
        ratio: range of aspect ratio of the origin aspect ratio to be cropped
        interpolation: interpolation method
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self._size = size
        else:
            self._size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ErrorValue("range should be of kind of (min, max)")

        self._interpolation = interpolation
        self._scale = scale
        self._ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = h * max(ratio)
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self._scale, self._ratio)
        assert _is_pil_image(img), 'image should be a PIL Image'
        img = crop(img, i, j, h, w)
        img = resize(img, self._size, self._interpolation)
        return img


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (tuple|int): Output size. If size is an int instead of a tuple
            like (h, w), a square crop (size, size) is made.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        return center_crop(img, self.size)
