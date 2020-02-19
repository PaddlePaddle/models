import math
import numpy as np
from pytracking.libs import fourier
from pytracking.libs import complex
from pytracking.libs.paddle_utils import _padding


def hann1d(sz: int, centered=True) -> np.ndarray:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - np.cos(
            (2 * math.pi / (sz + 2)) * np.arange(1, sz + 1, 1, 'float32')))
    w = 0.5 * (1 + np.cos(
        (2 * math.pi / (sz + 2)) * np.arange(0, sz // 2 + 1, 1, 'float32')))
    return np.concatenate([w, np.flip(w[1:sz - sz // 2], 0)])


def hann2d(sz: np.ndarray, centered=True) -> np.ndarray:
    """2D cosine window."""
    return np.reshape(hann1d(sz[0], centered), (1, 1, -1, 1)) * \
           np.reshape(hann1d(sz[1], centered), (1, 1, 1, -1))


def hann2d_clipped(sz: np.ndarray, effective_sz: np.ndarray,
                   centered=True) -> np.ndarray:
    """1D clipped cosine window."""

    # Ensure that the difference is even
    effective_sz += (effective_sz - sz) % 2
    effective_window = np.reshape(hann1d(effective_sz[0], True), (1, 1, -1, 1)) * \
                       np.reshape(hann1d(effective_sz[1], True), (1, 1, 1, -1))

    pad = np.int32((sz - effective_sz) / 2)
    window = _padding(
        effective_window, (pad[1], pad[1], pad[0], pad[0]), mode='replicate')

    if centered:
        return window
    else:
        mid = np.int32((sz / 2))
        window_shift_lr = np.concatenate(
            (window[..., mid[1]:], window[..., :mid[1]]), 3)
        return np.concatenate((window_shift_lr[..., mid[0]:, :],
                               window_shift_lr[..., :mid[0], :]), 2)


def gauss_fourier(sz: int, sigma: float, half: bool=False) -> np.ndarray:
    if half:
        k = np.arange(0, int(sz / 2 + 1), 1, 'float32')
    else:
        k = np.arange(-int((sz - 1) / 2), int(sz / 2 + 1), 1, 'float32')
    return (math.sqrt(2 * math.pi) * sigma / sz) * np.exp(-2 * np.square(
        math.pi * sigma * k / sz))


def gauss_spatial(sz, sigma, center=0, end_pad=0):
    k = np.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad, 1, 'float32')
    return np.exp(-1.0 / (2 * sigma**2) * np.square(k - center))


def label_function(sz: np.ndarray, sigma: np.ndarray):
    return np.reshape(gauss_fourier(sz[0], sigma[0]), (1, 1, -1, 1)) * \
           np.reshape(gauss_fourier(sz[1], sigma[1], True), (1, 1, 1, -1))


def label_function_spatial(sz: np.ndarray,
                           sigma: np.ndarray,
                           center: np.ndarray=None,
                           end_pad: np.ndarray=None):
    """The origin is in the middle of the image."""
    if center is None: center = np.zeros((2, ), 'float32')
    if end_pad is None: end_pad = np.zeros((2, ), 'float32')
    return np.reshape(gauss_spatial(sz[0], sigma[0], center[0], end_pad[0]), (1, 1, -1, 1)) * \
           np.reshape(gauss_spatial(sz[1], sigma[1], center[1], end_pad[1]), (1, 1, 1, -1))


def cubic_spline_fourier(f, a):
    """The continuous Fourier transform of a cubic spline kernel."""

    bf = (6 * (1 - np.cos(2 * math.pi * f)) + 3 * a * (1 - np.cos(4 * math.pi * f))
          - (6 + 8 * a) * math.pi * f * np.sin(2 * math.pi * f) - 2 * a * math.pi * f * np.sin(4 * math.pi * f)) \
         / (4 * math.pi ** 4 * f ** 4)
    bf[f == 0] = 1
    return bf


def get_interp_fourier(sz: np.ndarray,
                       method='ideal',
                       bicubic_param=0.5,
                       centering=True,
                       windowing=False,
                       device='cpu'):
    ky, kx = fourier.get_frequency_coord(sz)

    if method == 'ideal':
        interp_y = np.ones(ky.shape) / sz[0]
        interp_x = np.ones(kx.shape) / sz[1]
    elif method == 'bicubic':
        interp_y = cubic_spline_fourier(ky / sz[0], bicubic_param) / sz[0]
        interp_x = cubic_spline_fourier(kx / sz[1], bicubic_param) / sz[1]
    else:
        raise ValueError('Unknown method.')

    if centering:
        interp_y = complex.mult(interp_y,
                                complex.exp_imag((-math.pi / sz[0]) * ky))
        interp_x = complex.mult(interp_x,
                                complex.exp_imag((-math.pi / sz[1]) * kx))

    if windowing:
        raise NotImplementedError

    return interp_y, interp_x


def interpolate_dft(a: np.ndarray, interp_fs) -> np.ndarray:
    if isinstance(interp_fs, np.ndarray):
        return complex.mult(a, interp_fs)
    if isinstance(interp_fs, (tuple, list)):
        return complex.mult(complex.mult(a, interp_fs[0]), interp_fs[1])
    raise ValueError('"interp_fs" must be tensor or tuple of tensors.')


def max2d(a: np.ndarray) -> (np.ndarray, np.ndarray):
    """Computes maximum and argmax in the last two dimensions."""
    argmax_row = np.argmax(a, axis=-2)
    max_val_row = np.max(a, axis=-2)
    argmax_col = np.argmax(max_val_row, axis=-1)
    max_val = np.max(max_val_row, axis=-1)

    argmax_row = np.reshape(argmax_row, (
        argmax_col.size, -1))[np.arange(argmax_col.size), argmax_col.flatten()]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = np.concatenate(
        (np.expand_dims(argmax_row, -1), np.expand_dims(argmax_col, -1)), -1)

    return max_val, argmax
