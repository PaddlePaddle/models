# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import List, Optional, Tuple, Union

import paddle
from paddle import Tensor

from .signal import *
from .utils.error import ParameterError

EPS = 1e-10

__all_ = [
    'complex_norm',
    'magphase',
    'mel_to_hz',
    'hz_to_mel',
    'pad_center',
    'mel_frequencies',
    'fft_frequencies',
    'compute_fbank_matrix',
    'dft_matrix',
    'idft_matrix',
    'get_window',
    'power_to_db',
    'enframe'
    'deframe',
    'mu_encode',
    'mu_decode',
]


def complex_norm(x: Tensor) -> Tensor:
    """Compute compext norm of a given tensor.
    Typically,the input tensor is the result of a complex Fourier transform.

    Parameters:
        x: The input tensor of shape [..., 2]

    Returns:
        The element-wised l2-norm of input complex tensor.
    """
    if x.shape[-1] != 2:
        raise ParameterError(
            f'complex tensor must be of shape [..., 2], but received {x.shape} instead'
        )
    return paddle.sqrt(paddle.square(x).sum(axis=-1))


def magphase(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute compext norm of a given tensor.
    Typically,the input tensor is the result of a complex Fourier transform.

    Parameters:
        x: The input tensor of shape [..., 2]

    Returns:
        The element-wised l2-norm of input complex tensor.
    """
    if x.shape[-1] != 2:
        raise ParameterError(
            f'complex tensor must be of shape [..., 2], but received {x.shape} instead'
        )
    mag = paddle.sqrt(paddle.square(x).sum(axis=-1))
    x0 = x.reshape((-1, 2))
    phase = paddle.atan2(x0[:, 0], x0[:, 1])
    phase = phase.reshape(x.shape[:2])

    return mag, phase


def pad_center(data: Tensor,
               size: int,
               axis: int = -1,
               value: float = 0.0) -> Tensor:
    """Pad a tensor to a target length along a target axis.

    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`
    """
    padding_shape = data.shape
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    if lpad < 0:
        raise ParameterError(("Target size ({size:d}) must be "
                              "at least input size ({n:d})"))

    padding_shape[axis] = lpad
    padding = paddle.ones(padding_shape, data.dtype) * value
    padded_data = paddle.concat([padding, data, padding], axis)

    return padded_data


def hz_to_mel(freq: Union[Tensor, float], htk: bool = False) -> float:
    """Convert Hz to Mels.

    This function is consistent with librosa.hz_to_mel().
    """

    if htk:
        if isinstance(freq, Tensor):
            return 2595.0 * paddle.log10(1.0 + freq / 700.0)
        else:
            return 2595.0 * math.log10(1.0 + freq / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if isinstance(freq, Tensor):
        target = min_log_mel + paddle.log(
            freq / min_log_hz + 1e-10) / logstep  # prevent nan with 1e-10
        mask = (freq > min_log_hz).astype('float32')
        mels = target * mask + mels * (
            1 - mask)  # will replace by masked_fill OP in future
    else:
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz + 1e-10) / logstep

    return mels


def mel_to_hz(mel: Union[float, Tensor], htk: bool = False) -> Tensor:
    """Convert mel bin numbers to frequencies.

    This function is consistent with librosa.mel_to_hz().
    """
    if htk:
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mel
    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region
    if isinstance(mel, Tensor):
        target = min_log_hz * paddle.exp(logstep * (mel - min_log_mel))
        mask = (mel > min_log_mel).astype('float32')
        freqs = target * mask + freqs * (
            1 - mask)  # will replace by masked_fill OP in future
    else:
        if mel >= min_log_mel:
            freqs = min_log_hz * math.exp(logstep * (mel - min_log_mel))

    return freqs


def mel_frequencies(n_mels: int = 128,
                    fmin: float = 0.0,
                    fmax: float = 11025.0,
                    htk: bool = False):
    """Compute mel frequencies.

    This function is consistent with librosa.mel_frequencies().
    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)
    mels = paddle.linspace(min_mel, max_mel, n_mels)
    freqs = mel_to_hz(mels, htk=htk)
    return freqs


def fft_frequencies(sr: int, n_fft: int) -> Tensor:
    """Compute fourier frequencies.

    This function is consistent with librosa.fft_frequencies().
    """
    return paddle.linspace(0, float(sr) / 2, int(1 + n_fft // 2))


def compute_fbank_matrix(sr: int,
                         n_fft: int,
                         n_mels: int = 128,
                         fmin: float = 0.0,
                         fmax: Optional[float] = None,
                         htk: bool = False,
                         norm: str = 'slaney',
                         dtype: str = 'float32'):
    """Compute fbank matrix.

    This function is consistent with librosa.filters.mel().
    """
    if norm != 'slaney':
        raise ParameterError('norm must set to slaney')

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    weights = paddle.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = mel_f[1:] - mel_f[:-1]  #np.diff(mel_f)
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)
    #ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = paddle.maximum(paddle.zeros_like(lower),
                                    paddle.minimum(lower, upper))

    if norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm.unsqueeze(1)

    return weights


def dft_matrix(n: int, return_complex: bool = False) -> Tensor:
    """Compute dft matrix given dimension n.
    """
    x, y = paddle.meshgrid(paddle.arange(0, n), paddle.arange(0, n))
    z = x * y * (-2 * math.pi / n)
    cos = paddle.cos(z).unsqueeze(-1)
    sin = paddle.sin(z).unsqueeze(-1)
    if return_complex:
        return cos + paddle.to_tensor([1j]) * sin
    return paddle.concat([cos, sin], -1)


def idft_matrix(n: int, return_complex: bool = False) -> Tensor:
    """Compute inverse discrete Fourier transform matrix
    """

    x, y = paddle.meshgrid(paddle.arange(0, n), paddle.arange(0, n))
    z = x * y * (2 * math.pi / n)
    cos = paddle.cos(z).unsqueeze(-1)
    sin = paddle.sin(z).unsqueeze(-1)
    if return_complex:
        return cos + paddle.to_tensor([1j]) * sin
    return paddle.concat([cos, sin], -1)


def get_window(window: Union[str, Tuple[str, float]],
               win_length: int,
               fftbins: bool = True) -> Tensor:
    """Return a window of a given length and type.

    Parameters
        window : string, float, or tuple
            The type of window to create. See below for more details.
        win_length : int
            The number of samples in the window.
        fftbins : bool, optional
            If True (default), create a "periodic" window, ready to use with
            `ifftshift` and be multiplied by the result of an FFT
            If False, create a "symmetric" window, for use in filter design.

    Returns
       A window of length `win_length` and type `window`

    Notes
        This functional is consistent with scipy.signal.get_window
    """
    sym = not fftbins

    args = ()
    if isinstance(window, tuple):
        winstr = window[0]
        if len(window) > 1:
            args = window[1:]
    elif isinstance(window, str):
        if window in ['gaussian', 'exponential']:
            raise ValueError("The '" + window + "' window needs one or "
                             "more parameters -- pass a tuple.")
        else:
            winstr = window
    else:
        raise ValueError("%s as window type is not supported." %
                         str(type(window)))

    try:
        winfunc = eval(winstr + '_window')
    except KeyError as e:
        raise ValueError("Unknown window type.") from e

    params = (win_length, ) + args
    kwargs = {'sym': sym}
    return winfunc(*params, **kwargs)


def power_to_db(magnitude: Tensor,
                ref_value: float = 1.0,
                amin: float = 1e-10,
                top_db: Optional[float] = 80.0) -> Tensor:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(spect / ref)`` in a numerically
    stable way.

    Returns
        The spectrogram in log-scale
    Notes
        This function is consistent with librosa.
    """
    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if ref_value <= 0:
        raise ParameterError("ref_value must be strictly positive")

    ones = paddle.ones_like(magnitude)
    log_spec = 10.0 * paddle.log10(paddle.maximum(ones * amin, magnitude))
    log_spec -= 10.0 * math.log10(max(ref_value, amin))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = paddle.maximum(log_spec, ones * (log_spec.max() - top_db))

    return log_spec


def mu_encode(x: Tensor, mu: int = 255, quantized: bool = True) -> Tensor:
    """Mu-law encoding.
    Compute the mu-law decoding given an input code.
    When quantized is True, the result will be converted to
    integer in range [0,mu-1]. Otherwise, the resulting signal
    is in range [-1,1]


    Reference:
        https://en.wikipedia.org/wiki/%CE%9C-law_algorithm

    """
    mu = 255
    y = paddle.sign(x) * paddle.log1p(mu * paddle.abs(x)) / math.log1p(mu)
    if quantized:
        y = (y + 1) / 2 * mu + 0.5  # convert to [0 , mu-1]
        y = paddle.clip(y, min=0, max=mu).astype('int32')
    return y


def mu_decode(y: Tensor, mu: int = 255, quantized: bool = True) -> Tensor:
    """Mu-law decoding.
    Compute the mu-law decoding given an input code.

    This function assumes that the input y is in the
    range [0,mu-1] when quantize is True and [-1,1] otherwise

    Reference:
        https://en.wikipedia.org/wiki/%CE%9C-law_algorithm

    """
    if mu < 1:
        raise ParameterError('mu is typically set as 2**k-1, k=1, 2, 3,...')

    mu = mu - 1
    if quantized:  # undo the quantization
        y = y * 2 / mu - 1
    x = paddle.sign(y) / mu * ((1 + mu)**paddle.abs(y) - 1)
    return x


def enframe(signal: Tensor, hop_length: int, win_length: int) -> Tensor:

    raise NotImplementedError()


def deframe(frames: Tensor,
            n_fft: int,
            hop_length: int,
            win_length: int,
            signal_length: Optional[int] = None):
    """Unpack audio frames into audio singal. The frames are typically the output of inverse STFT that
    needs to be converted back to audio signals.

    This function is implemented by transposing and reshaping.
    """
    assert frames.ndim == 2 or frames.ndim == 3, f'The input frame must be a 2-d or 3-d tensor,but received ndim={frames.ndim} instead'
    if frames.ndim == 2:
        frames = frames.unsqueeze(0)
    assert n_fft == frames.shape[
        1], f'n_fft must be the same as the fraquency dimension of frames, but received {n_fft}!={frames.shape[1]}'
    frame_num = frames.shape[-1]
    overlap = (win_length - hop_length) // 2
    signal = paddle.zeros((
        frames.shape[0],
        hop_length * frame_num,
    ))
    start = n_fft // 2 - win_length // 2
    signal = frames[:, start + overlap:start + win_length - overlap, :]
    signal = signal.transpose((0, 2, 1))
    signal = signal.reshape((frames.shape[0], -1))
    if signal_length is None:
        return signal
    else:
        diff = signal.shape[-1] - signal_length
        return signal[:, diff // 2:-diff // 2]
