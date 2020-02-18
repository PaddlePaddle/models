import random
import numpy as np
import math
import cv2 as cv
from paddle.fluid import layers
from pytracking_pp.libs.paddle_utils import PTensor


class Transform:
    """ Class for applying various image transformations."""

    def __call__(self, *args):
        rand_params = self.roll()
        if rand_params is None:
            rand_params = ()
        elif not isinstance(rand_params, tuple):
            rand_params = (rand_params,)
        output = [self.transform(img, *rand_params) for img in args]
        if len(output) == 1:
            return output[0]
        return output

    def roll(self):
        return None

    def transform(self, img, *args):
        """Must be deterministic"""
        raise NotImplementedError


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            if not isinstance(args, tuple):
                args = (args,)
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = np.reshape(mean, [-1, 1, 1])
        self.std = np.reshape(std, [-1, 1, 1])

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return (tensor - self.mean) / self.std


class ToArray(Transform):
    """ Transpose image and jitter brightness"""

    def __init__(self, brightness_jitter=0.0):
        self.brightness_jitter = brightness_jitter

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return img.astype('float32') / 255.


class ToArrayAndJitter(Transform):
    """ Transpose image and jitter brightness"""

    def __init__(self, brightness_jitter=0.0):
        self.brightness_jitter = brightness_jitter

    def roll(self):
        return np.random.uniform(max(0, 1 - self.brightness_jitter), 1 + self.brightness_jitter)

    def transform(self, img, brightness_factor):
        # handle numpy array
        img = img.transpose((2, 0, 1))

        # backward compatibility
        return np.clip(img.astype('float32') * brightness_factor / 255.0, 0.0, 1.0)


class ToGrayscale(Transform):
    """Converts image to grayscale with probability"""

    def __init__(self, probability=0.5):
        self.probability = probability
        self.color_weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def roll(self):
        return random.random() < self.probability

    def transform(self, img, do_grayscale):
        if do_grayscale:
            if isinstance(img, PTensor):
                raise NotImplementedError('Implement paddle variant.')
            img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            return np.stack([img_gray, img_gray, img_gray], axis=2)
            # return np.repeat(np.sum(img * self.color_weights, axis=2, keepdims=True).astype(np.uint8), 3, axis=2)
        return img


class RandomHorizontalFlip(Transform):
    """Horizontally flip the given NumPy Image randomly with a probability p."""

    def __init__(self, probability=0.5):
        self.probability = probability

    def roll(self):
        return random.random() < self.probability

    def transform(self, img, do_flip):
        if do_flip:
            if isinstance(img, PTensor):
                return layers.reverse(img, 2)
            return np.fliplr(img).copy()
        return img
