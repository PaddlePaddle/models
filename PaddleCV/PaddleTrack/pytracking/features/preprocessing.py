import numpy as np
import cv2 as cv
from paddle.fluid import dygraph
from paddle.fluid import layers
from pytracking.libs.paddle_utils import PTensor, n2p, _padding, squeeze, unsqueeze


def numpy_to_paddle(a: np.ndarray):
    return unsqueeze(
        layers.transpose(
            layers.cast(dygraph.to_variable(a), 'float32'), [2, 0, 1]), [0])


def paddle_to_numpy(a: PTensor):
    return layers.transpose(squeeze(a, [0]), [1, 2, 0]).numpy()


def sample_patch(im: np.ndarray,
                 pos: np.ndarray,
                 sample_sz: np.ndarray,
                 output_sz: np.ndarray=None):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
    """

    # copy and convert
    posl = pos.astype('long')

    # Compute pre-downsampling factor
    if output_sz is not None:
        resize_factor = np.min(
            sample_sz.astype('float32') / output_sz.astype('float32'))
        df = int(max(int(resize_factor - 0.1), 1))
    else:
        df = int(1)

    sz = sample_sz.astype('float32') / df  # new size

    # Do downsampling
    if df > 1:
        os = posl % df  # offset
        posl = ((posl - os) / df).astype('long')  # new position
        im2 = im[os[0]::df, os[1]::df]  # downsample
    else:
        im2 = im

    # compute size to crop
    szl = np.maximum(
        np.round(sz), np.array(
            [2., 2.], dtype='float32')).astype('long')

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl // 2

    # Get image patch
    im_patch = _padding(
        im2, (0, 0, -tl[1], br[1] - im2.shape[1] + 1, -tl[0],
              br[0] - im2.shape[0] + 1),
        mode='replicate')

    if output_sz is None or (im_patch.shape[0] == output_sz[0] and
                             im_patch.shape[1] == output_sz[1]):
        return im_patch

    # Resample
    osz = output_sz.astype('long')
    im_patch = cv.resize(
        im_patch, (osz[1], osz[0]), interpolation=cv.INTER_LINEAR)
    return im_patch


def sample_patch_with_mean_pad(im: np.ndarray,
                               pos: np.ndarray,
                               sample_sz: np.ndarray,
                               output_sz: np.ndarray=None):
    """Sample an image patch.

    args:
        im: Image
        pos: center position of crop
        sample_sz: size to crop
        output_sz: size to resize to
    """

    # copy and convert
    # posl = np.round(pos).astype('long')  # TODO: maybe we should use round
    posl = pos.astype('long')

    im2 = im
    sz = sample_sz.astype('float32')
    # compute size to crop
    szl = np.maximum(
        np.round(sz), np.array(
            [2., 2.], dtype='float32')).astype('long')

    # Extract top and bottom coordinates
    tl = posl - (szl - 1) // 2
    br = posl + szl // 2

    # Get image patch
    im_patch = _padding(
        im2, (0, 0, -tl[1], br[1] - im2.shape[1] + 1, -tl[0],
              br[0] - im2.shape[0] + 1),
        mode='replicate')

    if output_sz is None or (im_patch.shape[0] == output_sz[0] and
                             im_patch.shape[1] == output_sz[1]):
        return im_patch

    # Resample
    osz = output_sz.astype('long')
    im_patch = cv.resize(
        im_patch, (osz[1], osz[0]), interpolation=cv.INTER_LINEAR)
    return im_patch
