import numpy as np
import math

from paddle.fluid import layers

import cv2 as cv

from pytracking.features.preprocessing import numpy_to_paddle, paddle_to_numpy
from pytracking.libs.Fconv2d import FConv2D
from pytracking.libs.paddle_utils import PTensor, _padding, n2p


class Transform:
    """Base data augmentation transform class."""

    def __init__(self, output_sz=None, shift=None):
        self.output_sz = output_sz
        self.shift = (0, 0) if shift is None else shift

    def __call__(self, image):
        raise NotImplementedError

    def crop_to_output(self, image, shift=None):
        if isinstance(image, PTensor):
            imsz = image.shape[2:]
        else:
            imsz = image.shape[:2]

        if self.output_sz is None:
            pad_h = 0
            pad_w = 0
        else:
            pad_h = (self.output_sz[0] - imsz[0]) / 2
            pad_w = (self.output_sz[1] - imsz[1]) / 2
        if shift is None:
            shift = self.shift
        pad_left = math.floor(pad_w) + shift[1]
        pad_right = math.ceil(pad_w) - shift[1]
        pad_top = math.floor(pad_h) + shift[0]
        pad_bottom = math.ceil(pad_h) - shift[0]

        if isinstance(image, PTensor):
            return _padding(
                image, (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate')
        else:
            return _padding(
                image, (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate')


class Identity(Transform):
    """Identity transformation."""

    def __call__(self, image):
        return self.crop_to_output(image)


class FlipHorizontal(Transform):
    """Flip along horizontal axis."""

    def __call__(self, image):
        if isinstance(image, PTensor):
            return self.crop_to_output(layers.reverse(image, 3))
        else:
            return self.crop_to_output(np.fliplr(image))


class FlipVertical(Transform):
    """Flip along vertical axis."""

    def __call__(self, image: PTensor):
        if isinstance(image, PTensor):
            return self.crop_to_output(layers.reverse(image, 2))
        else:
            return self.crop_to_output(np.flipud(image))


class Translation(Transform):
    """Translate."""

    def __init__(self, translation, output_sz=None, shift=None):
        super().__init__(output_sz, shift)
        self.shift = (self.shift[0] + translation[0],
                      self.shift[1] + translation[1])

    def __call__(self, image):
        return self.crop_to_output(image)


class Scale(Transform):
    """Scale."""

    def __init__(self, scale_factor, output_sz=None, shift=None):
        super().__init__(output_sz, shift)
        self.scale_factor = scale_factor

    def __call__(self, image):
        # Calculate new size. Ensure that it is even so that crop/pad becomes easier
        h_orig, w_orig = image.shape[2:]

        if h_orig != w_orig:
            raise NotImplementedError

        h_new = round(h_orig / self.scale_factor)
        h_new += (h_new - h_orig) % 2
        w_new = round(w_orig / self.scale_factor)
        w_new += (w_new - w_orig) % 2

        if isinstance(image, PTensor):
            image_resized = layers.resize_bilinear(
                image, [h_new, w_new], align_corners=False)
        else:
            image_resized = cv.resize(
                image, (w_new, h_new), interpolation=cv.INTER_LINEAR)
        return self.crop_to_output(image_resized)


class Affine(Transform):
    """Affine transformation."""

    def __init__(self, transform_matrix, output_sz=None, shift=None):
        super().__init__(output_sz, shift)
        self.transform_matrix = transform_matrix

    def __call__(self, image, crop=True):
        if isinstance(image, PTensor):
            return self.crop_to_output(
                numpy_to_paddle(self(
                    paddle_to_numpy(image), crop=False)))
        else:
            warp = cv.warpAffine(
                image,
                self.transform_matrix,
                image.shape[1::-1],
                borderMode=cv.BORDER_REPLICATE)
            if crop:
                return self.crop_to_output(warp)
            else:
                return warp


class Rotate(Transform):
    """Rotate with given angle."""

    def __init__(self, angle, output_sz=None, shift=None):
        super().__init__(output_sz, shift)
        self.angle = math.pi * angle / 180

    def __call__(self, image, crop=True):
        if isinstance(image, PTensor):
            return self.crop_to_output(
                numpy_to_paddle(self(
                    paddle_to_numpy(image), crop=False)))
        else:
            c = (np.expand_dims(np.array(image.shape[:2]), 1) - 1) / 2
            R = np.array([[math.cos(self.angle), math.sin(self.angle)],
                          [-math.sin(self.angle), math.cos(self.angle)]])
            H = np.concatenate([R, c - R @c], 1)
            warp = cv.warpAffine(
                image, H, image.shape[1::-1], borderMode=cv.BORDER_REPLICATE)
            if crop:
                return self.crop_to_output(warp)
            else:
                return warp


class Blur(Transform):
    """Blur with given sigma (can be axis dependent)."""

    def __init__(self, sigma, output_sz=None, shift=None):
        super().__init__(output_sz, shift)
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2 * s) for s in self.sigma]

        x_coord = [
            np.arange(
                -sz, sz + 1, 1, dtype='float32') for sz in self.filter_size
        ]
        self.filter_np = [
            np.exp(0 - (x * x) / (2 * s**2))
            for x, s in zip(x_coord, self.sigma)
        ]
        self.filter_np[0] = np.reshape(
            self.filter_np[0], [1, 1, -1, 1]) / np.sum(self.filter_np[0])
        self.filter_np[1] = np.reshape(
            self.filter_np[1], [1, 1, 1, -1]) / np.sum(self.filter_np[1])

    def __call__(self, image):
        if isinstance(image, PTensor):
            sz = image.shape[2:]
            filter = [n2p(f) for f in self.filter_np]
            im1 = FConv2D(
                layers.reshape(image, [-1, 1, sz[0], sz[1]]),
                filter[0],
                padding=(self.filter_size[0], 0))
            return self.crop_to_output(
                layers.reshape(
                    FConv2D(
                        im1, filter[1], padding=(0, self.filter_size[1])),
                    [1, -1, sz[0], sz[1]]))
        else:
            return paddle_to_numpy(self(numpy_to_paddle(image)))
