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

from .core import *
from .utils import ParameterError

EPS = 1e-10

__all_ = [
    'dft_matrix',
    'idft_matrix',
    'stft',
    'istft',
    'spectrogram',
    'melspectrogram',
    'complex_norm',
    'magphase',
    'mel_to_hz',
    'hz_to_mel',
    'mel_frequencies',
    'fft_frequencies',
    'compute_fbank_matrix',
    'get_window',
    'power_to_db',
    'enframe',
    'deframe',
    'mu_law_encode',
    'mu_law_decode',
    'random_masking',
    'random_cropping',
    'center_padding',
    'dct_matrx',
    'mfcc',
]


def _randint(n):
    """The helper function for computing randint.
    """
    return int(paddle.randint(n))


def complex_norm(x: Tensor) -> Tensor:
    """Compute compext norm of a given tensor.
    Typically, the input tensor is the result of a complex Fourier transform.
    Parameters:
        x(Tensor): The input tensor of shape (..., 2)

    Returns:
        The element-wised l2-norm of input complex tensor.
     Examples:

        .. code-block:: python

        x = paddle.rand((32, 16000))
        y = F.stft(x, n_fft=512)
        z = F.complex_norm(y)
        print(z.shape)
        >> [32, 257, 126]

    """
    if x.shape[-1] != 2:
        raise ParameterError(
            f'complex tensor must be of shape (..., 2), but received {x.shape} instead'
        )
    return paddle.sqrt(paddle.square(x).sum(axis=-1))


def magphase(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute compext norm of a given tensor.
    Typically,the input tensor is the result of a complex Fourier transform.
    Parameters:
        x(Tensor): The input tensor of shape (..., 2).
    Returns:
        The tuple of magnitude and phase.

    Shape:
        x: the shape of x is arbitrary, with the shape of last axis being 2
        outputs: the shapes of magnitude and phase are both input.shape[:-1]

     Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        x = paddle.randn((10, 10, 2))
        angle, phase = F.magphase(x)

    """
    if x.shape[-1] != 2:
        raise ParameterError(
            f'complex tensor must be of shape (..., 2), but received {x.shape} instead'
        )
    mag = paddle.sqrt(paddle.square(x).sum(axis=-1))
    x0 = x.reshape((-1, 2))
    phase = paddle.atan2(x0[:, 0], x0[:, 1])
    phase = phase.reshape(x.shape[:-1])

    return mag, phase


def hz_to_mel(freq: Union[Tensor, float],
              htk: bool = False) -> Union[Tensor, float]:
    """Convert Hz to Mels.

    Parameters:
        freq: the input tensor of arbitrary shape, or a single floating point number.
        htk: use HTK formula to do the conversion.
            The default value is False.
    Returns:
        The frequencies represented in Mel-scale.
    Notes:
        This function is consistent with librosa.hz_to_mel().

     Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        print(F.hz_to_mel(10))
        >> 10
        print(F.hz_to_mel(paddle.to_tensor([0, 100, 1600])))
        >> Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                [0., 1.50000000, 21.83624077])

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


def mel_to_hz(mel: Union[float, Tensor],
              htk: bool = False) -> Union[float, Tensor]:
    """Convert mel bin numbers to frequencies.

    Parameters:
        mel: the mel frequency represented as a tensor of arbitrary shape, or a floating point number.
        htk: use HTK formula to do the conversion.
    Returns:
        The frequencies represented in hz.
    Notes:
        This function is consistent with librosa.mel_to_hz().

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        print(F.mel_to_hz(10))
        >> 666.6666666666667
        print(F.mel_to_hz(paddle.to_tensor([0, 1.0, 10.0])))
        >> Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                [0., 66.66666412, 666.66662598])

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
        mask = (mel > min_log_mel).astype(mel.dtype)
        freqs = target * mask + freqs * (
            1 - mask)  # will replace by masked_fill OP in future
    else:
        if mel >= min_log_mel:
            freqs = min_log_hz * math.exp(logstep * (mel - min_log_mel))

    return freqs


def mel_frequencies(n_mels: int = 128,
                    f_min: float = 0.0,
                    f_max: float = 11025.0,
                    htk: bool = False,
                    dtype: str = 'float64') -> Tensor:
    """Compute mel frequencies.

    Parameters:
        n_mels(int): number of Mel bins.
        f_min(float): the lower cut-off frequency, below which the filter response is zero.
        f_max(float): the upper cut-off frequency, above which the filter response is zero.
        htk(bool): whether to use htk formula.
        dtype(str): the datatype of the return frequencies.

    Returns:
        The frequencies represented in Mel-scale

    Notes:
        This function is consistent with librosa.mel_frequencies().

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        print(F.mel_frequencies(8))
        >> Tensor(shape=[8], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                [0., 475.33898926, 950.67797852, 1551.68481445, 2533.36230469,
                4136.09960938, 6752.81396484, 11024.99902344])

    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(f_min, htk=htk)
    max_mel = hz_to_mel(f_max, htk=htk)
    mels = paddle.linspace(min_mel, max_mel, n_mels, dtype=dtype)
    freqs = mel_to_hz(mels, htk=htk)
    return freqs


def fft_frequencies(sr: int, n_fft: int, dtype: str = 'float64') -> Tensor:
    """Compute fourier frequencies.

    Parameters:
        sr(int): the audio sample rate.
        n_fft(float): the number of fft bins.
        dtype(str): the datatype of the return frequencies.
    Returns:
        The frequencies represented in hz.
    Notes:
        This function is consistent with librosa.fft_frequencies().

    Examples:
        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        print(F.fft_frequencies(16000, 512))
        >> Tensor(shape=[257], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                [0., 31.25000000, 62.50000000, ...]

    """
    return paddle.linspace(0, float(sr) / 2, int(1 + n_fft // 2), dtype=dtype)


def compute_fbank_matrix(sr: int,
                         n_fft: int,
                         n_mels: int = 128,
                         f_min: float = 0.0,
                         f_max: Optional[float] = None,
                         htk: bool = False,
                         norm: Union[str, float] = 'slaney',
                         dtype: str = 'float64') -> Tensor:
    """Compute fbank matrix.

    Parameters:
        sr(int): the audio sample rate.
        n_fft(int): the number of fft bins.
        n_mels(int): the number of Mel bins.
        f_min(float): the lower cut-off frequency, below which the filter response is zero.
        f_max(float): the upper cut-off frequency, above which the filter response is zero.
        htk: whether to use htk formula.
        return_complex(bool): whether to return complex matrix. If True, the matrix will
            be complex type. Otherwise, the real and image part will be stored in the last
            axis of returned tensor.
        dtype(str): the datatype of the returned fbank matrix.

    Returns:
        The fbank matrix of shape (n_mels, int(1+n_fft//2)).
    Shape:
        output: (n_mels, int(1+n_fft//2))
    Notes:
        This function is consistent with librosa.filters.mel().

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        m = F.compute_fbank_matrix(16000, 512)
        print(m.shape)
        >>[128, 257]

    """

    if f_max is None:
        f_max = float(sr) / 2

    # Initialize the weights
    weights = paddle.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft, dtype=dtype)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2,
                            f_min=f_min,
                            f_max=f_max,
                            htk=htk,
                            dtype=dtype)

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

    # Slaney-style mel is scaled to be approx constant energy per channel
    if norm == 'slaney':
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm.unsqueeze(1)
    elif isinstance(norm, int) or isinstance(norm, float):
        weights = paddle.nn.functional.normalize(weights, p=norm, axis=-1)

    return weights


def dft_matrix(n: int,
               return_complex: bool = False,
               dtype: str = 'float64') -> Tensor:
    """Compute discrete Fourier transform matrix.

    Parameters:
        n(int): the size of dft matrix.
        return_complex(bool): whether to return complex matrix. If True, the matrix will
            be complex type. Otherwise, the real and image part will be stored in the last
            axis of returned tensor.
        dtype(str): the datatype of the returned dft matrix.

    Shape:
        output: [n, n] or [n,n,2]

    Returns:
        Complex tensor of shape (n,n) if return_complex=True, and of shape (n,n,2) otherwise.

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        m = F.dft_matrix(512)
        print(m.shape)
        >> [512, 512, 2]
        m = F.dft_matrix(512, return_complex=True)
        print(m.shape)
        >> [512, 512]

    """
    # This is due to a bug in paddle in lacking support for complex128, as of paddle 2.1.0
    if return_complex and dtype == 'float64':
        raise ValueError('not implemented')

    x, y = paddle.meshgrid(paddle.arange(0, n), paddle.arange(0, n))
    z = x.astype(dtype) * y.astype(dtype) * paddle.to_tensor(
        (-2 * math.pi / n), dtype)
    cos = paddle.cos(z)
    sin = paddle.sin(z)

    if return_complex:
        return cos + paddle.to_tensor([1j]) * sin
    cos = cos.unsqueeze(-1)
    sin = sin.unsqueeze(-1)
    return paddle.concat([cos, sin], -1)


def idft_matrix(n: int,
                return_complex: bool = False,
                dtype: str = 'float64') -> Tensor:
    """Compute inverse discrete Fourier transform matrix

    Parameters:
        n(int): the size of idft matrix.
        return_complex(bool): whether to return complex matrix. If True, the matrix will
            be complex type. Otherwise, the real and image part will be stored in the last
            axis of returned tensor.
        dtype(str): the data type of returned idft matrix.
    Returns:
        Complex tensor of shape (n,n) if return_complex=True, and of shape (n,n,2) otherwise.
    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        m = F.dft_matrix(512)
        print(m.shape)
        >> [512, 512, 2]
        m = F.dft_matrix(512, return_complex=True)
        print(m.shape)
        >> [512, 512]

    """

    if return_complex and dtype == 'float64':  # there is a bug in paddle for complex128 datatype
        raise ValueError('not implemented')

    x, y = paddle.meshgrid(paddle.arange(0, n, dtype=dtype),
                           paddle.arange(0, n, dtype=dtype))
    z = x.astype(dtype) * y.astype(dtype) * paddle.to_tensor(
        (2 * math.pi / n), dtype)
    cos = paddle.cos(z)
    sin = paddle.sin(z)
    if return_complex:
        return cos + paddle.to_tensor([1j]) * sin
    cos = cos.unsqueeze(-1)
    sin = sin.unsqueeze(-1)
    return paddle.concat([cos, sin], -1)


def dct_matrix(n_mfcc: int,
               n_mels: int,
               dct_norm: Optional[str] = 'ortho',
               dtype: str = 'float64') -> Tensor:
    """Compute discrete cosine transform (DCT) matrix used in MFCC computation.

    Parameters:
        n_mfcc(int): the number of coefficients in MFCC.
        n_mels(int): the number of mel bins in the melspectrogram tranform preceding MFCC.
        dct_norm(None|str): the normalization of the dct transform. If 'ortho', use the orthogonal normalization.
            If None, not normalization is applied. Default: 'ortho'.
        dtype(str): the data type of returned dct matrix.

    Shape:
        output: [n_mels,n_mfcc]

    Returns:
        The dct matrix of shape [n_mels,n_mfcc]

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        m = F.dct_matrix(n_mfcc=20,n_mels=64)
        print(m.shape)
        >> [64, 20]

    """
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = paddle.arange(float(n_mels), dtype=dtype)
    k = paddle.arange(float(n_mfcc), dtype=dtype).unsqueeze(1)
    dct = paddle.cos(math.pi / float(n_mels) * (n + 0.5) *
                     k)  # size (n_mfcc, n_mels)
    if dct_norm is None:
        dct *= 2.0
    else:
        assert dct_norm == "ortho"
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


def get_window(window: Union[str, Tuple[str, float]],
               win_length: int,
               fftbins: bool = True,
               dtype: str = 'float64') -> Tensor:
    """Return a window of a given length and type.
    Parameters:
        window(str|(str,float)): the type of window to create.
        win_length(int): the number of samples in the window.
        fftbins(bool): If True, create a "periodic" window. Otherwise,
            create a "symmetric" window, for use in filter design.
    Returns:
       The window represented as a tensor.
    Notes:
        This functional is consistent with scipy.signal.get_window()
    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        w = F.get_window('hann', win_length=128)
        print(w.shape)
        >> [128]

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
    return winfunc(*params, dtype=dtype, **kwargs)


def power_to_db(magnitude: Tensor,
                ref_value: float = 1.0,
                amin: float = 1e-10,
                top_db: Optional[float] = 80.0) -> Tensor:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units.
    The function computes the scaling ``10 * log10(x / ref)`` in a numerically
    stable way.

    Parameters:
        magnitude(Tensor): the input magnitude tensor of any shape.
        ref_value(float): the reference value. If smaller than 1.0, the db level
            of the signal will be pulled up accordingly. Otherwise, the db level
            is pushed down.
        amin(float): the minimum value of input magnitude, below which the input
            magnitude is clipped(to amin).
        top_db(float): the maximum db value of resulting spectrum, above which the
            spectrum is clipped(to top_db).
    Returns:
        The spectrogram in log-scale.
    shape:
        input: any shape
        output: same as input
    Notes:
        This function is consistent with librosa.power_to_db().
    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        F.power_to_db(paddle.rand((10, 10)))
        >> Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                [[-6.22858429, -3.51512218],
                [-0.38168561, -1.44466150]])

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


def mu_law_encode(x: Tensor, mu: int = 256, quantized: bool = True) -> Tensor:
    """Mu-law encoding.
    Compute the mu-law decoding given an input code.
    When quantized is True, the result will be converted to
    integer in range [0,mu-1]. Otherwise, the resulting signal
    is in range [-1,1]

    Parameters:
        x(Tensor): the input tensor of arbitrary shape to be encoded.
        mu(int): the maximum value (depth) of encoded signal. The signal will be
        clip to be in range [0,mu-1].
        quantized(bool): indicate whether the signal will quantized to integers.

    Examples:
        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        F.mu_law_encode(paddle.randn((2, 8)))
        >> Tensor(shape=[2, 8], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
                [[0, 5, 30, 255, 255, 255, 12, 13],
                [0, 241, 8, 243, 7, 35, 84, 228]])

    Reference:
        https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    """
    mu = mu - 1
    y = paddle.sign(x) * paddle.log1p(mu * paddle.abs(x)) / math.log1p(mu)
    if quantized:
        y = (y + 1) / 2 * mu + 0.5  # convert to [0 , mu-1]
        y = paddle.clip(y, min=0, max=mu).astype('int32')
    return y


def mu_law_decode(x: Tensor, mu: int = 256, quantized: bool = True) -> Tensor:
    """Mu-law decoding.
    Compute the mu-law decoding given an input code.

    Parameters:
        x(Tensor): the input tensor of arbitrary shape to be decoded.
        mu(int): the maximum value of encoded signal, which should be the
        same as that in mu_law_encode().

        quantized(bool): whether the signal has been quantized to integers.
        The value should be the same as that used in mu_law_encode()
    shape:
        input: any shape
        output: same as input

    Notes:
        This function assumes that the input x is in the
        range [0,mu-1] when quantize is True and [-1,1] otherwise.



    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        F.mu_law_decode(paddle.randint(0, 255, shape=(2, 8)))
        >> Tensor(shape=[2, 8], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                [[0.00796641, -0.28048742, -0.13789690,  0.67482352, -0.05550348, -0.00377374,  0.64593655,  0.03134083],
                [0.45497340, -0.29312974,  0.29312995, -0.70499402,  0.51892924, -0.15078513,  0.07322186,  0.70499456]])

    Reference:
        https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    """
    if mu < 1:
        raise ParameterError('mu is typically set as 2**k-1, k=1, 2, 3,...')

    mu = mu - 1
    if quantized:  # undo the quantization
        x = x * 2 / mu - 1
    x = paddle.sign(x) / mu * ((1 + mu)**paddle.abs(x) - 1)
    return x


def enframe(signal: Tensor, hop_length: int, win_length: int) -> Tensor:
    raise NotImplementedError()


def deframe(frames: Tensor,
            n_fft: int,
            hop_length: int,
            win_length: int,
            signal_length: Optional[int] = None):
    """Unpack audio frames into audio singal.
    The frames are typically the output of inverse STFT that needs to be converted back to audio signals.

    Parameters:
        frames(Tensor): the input audio frames of shape (N,n_fft,frame_number) or (n_fft,frame_number)
        The frames are typically obtained from the output of inverse STFT.
        n_fft(int): the number of fft bins, see paddleaudio.functional.stft()
        hop_length(int): the hop length, see paddleaudio.functional.stft()
        win_length(int): the window length, see paddleaudio.functional.stft()
        signal_length(int): the original signal length. If None, the resulting
        signal length is determined by hop_length*win_length. Otherwised, the signal is
        centrally cropped to signal_length.
    Returns:
        Tensor: the unpacked signal.
    Notes:
        This function is implemented by transposing and reshaping.

     Shape:
        - input:  (N,n_fft,frame_number] or (n_fft,frame_number)
        - output: ( N, signal_length)

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        x = paddle.rand((128, 200))
        x = F.deframe(x, n_fft=128, hop_length=64, win_length=200)
        print(x.shape)
        >> [128, 200]

    """
    assert frames.ndim == 2 or frames.ndim == 3, (
        f'The input frame must be a 2-d or 3-d tensor, ' +
        f'but received ndim={frames.ndim} instead')
    if frames.ndim == 2:
        frames = frames.unsqueeze(0)
    assert n_fft == frames.shape[1], (
        f'n_fft must be the same as the frequency dimension of frames, ' +
        f'but received {n_fft}!={frames.shape[1]}')

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
    if signal_length is None or signal_length == signal.shape[-1]:
        return signal
    else:
        assert signal_length < signal.shape[-1], (
            'signal_length must be smaller than hop_length*win_length, ' +
            f'but received signal_length={signal_length}, ' +
            f'hop_length*win_length={hop_length*win_length}')
        diff = signal.shape[-1] - signal_length
        return signal[:, diff // 2:-diff // 2]


def random_masking(x: Tensor,
                   max_mask_count: int,
                   max_mask_width: int,
                   axis: int = -1) -> Tensor:
    """Apply random masking to a given input tensor x along axis.
    The function randomly mask input x with zeros along axis. The maximum number of masking regions
    is defined by max_mask_count, each of which has maximum zero-out width defined by max_mask_width.

    Parameters:
        x(Tensor): The maximum number of masking regions.
        max_mask_count(int): the maximum number of masking regions.
        max_mask_width(int)：the maximum zero-out width of each region.
        axis(int): the axis along which to apply masking.
            The default value is -1.
    Returns:
        Tensor: the tensor after masking.
    Examples:

        .. code-block:: python

        x = paddle.rand((64, 100))
        x = F.random_masking(x, max_mask_count=10, max_mask_width=2, axis=0)
        print((x[:, 0] == 0).astype('int32').sum())
        >> Tensor(shape=[1], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
                [5])

        x = paddle.rand((64, 100))
        x = F.random_masking(x, max_mask_count=10, max_mask_width=2, axis=1)
        print((x[0, :] == 0).astype('int32').sum())
        >> Tensor(shape=[1], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
                [8])

    """

    assert x.ndim == 2 or x.ndim, (f'only supports 2d or 3d tensor, ' +
                                   f'but received ndim={x.ndim}')

    if x.ndim == 2:
        x = x.unsqueeze(0)  #extend for batching
        squeeze = True
        if axis != -1:
            assert axis in [
                0, 1, -1
            ], (f'mask axis must be in [0,1,-1] for 2d tensor input, ' +
                f'but received {axis}')
            axis += 1
    else:
        squeeze = False
        assert axis in [1, 2, -1,
                        -2], ('mask axis must be in [1,2,-1,-2] ' +
                              f'for 3d tensor input,but received {axis}')

    zero_tensor = paddle.to_tensor(0.0, dtype=x.dtype)

    n = x.shape[axis]
    num_masks = _randint(max_mask_count + 1)
    mask_width = _randint(max_mask_width) + 1

    if axis == 1:
        for _ in range(num_masks):
            start = _randint(n - mask_width)
            x[:, start:start + mask_width, :] = zero_tensor
    else:
        for _ in range(num_masks):
            start = _randint(n - mask_width)
            x[:, :, start:start + mask_width] = zero_tensor
    if squeeze:
        x = x.squeeze()
    return x


def random_cropping(x: Tensor, target_size: int, axis=-1) -> Tensor:
    """Randomly crops input tensor x along given axis.
    The function randomly crops input x to target_size along axis, such that output.shape[axis] == target_size

    Parameters:
        x(Tensor): the input tensor to apply random cropping.
        target_size(int): the target length after cropping.
        axis(int)：the axis along which to apply cropping.
    Returns:
        Tensor: the cropped tensor. If target_size >= x.shape[axis], the original input tensor is returned
        without cropping.
    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        x = paddle.randn((2, 8))
        y = F.random_cropping(x, target_size=6)
        print(y.shape)
        >> [2, 6]
        y = F.random_cropping(x, target_size=10)
        print(y.shape)
        >> [2, 8]  # same as x

    """

    assert axis < x.ndim, ('axis must be smaller than x.ndim, ' +
                           f'but received aixs={axis},x.ndim={x.ndim}')

    assert x.ndim in [1, 2, 3], ('only accept 1d/2d/3d tensor, ' +
                                 f'but received x.ndim={x.ndim}')

    shape = x.shape
    if target_size >= shape[axis]:
        return x  # nothing to do

    start = _randint(shape[axis] - target_size)
    axes = [i for i in range(x.ndim)]
    starts = [0 for i in range(x.ndim)]
    ends = [shape[i] for i in range(x.ndim)]
    starts[axis] = start
    ends[axis] = start + target_size
    return paddle.slice(x, axes, starts, ends)


def center_padding(x: Tensor,
                   target_size: int,
                   axis: int = -1,
                   pad_value: float = 0.0) -> Tensor:
    """Centrally pad input tensor x along given axis.
    The function pads input x with pad_value to target_size along axis, such that output.shape[axis] == target_size

    Parameters:
        x(Tensor): the input tensor to apply padding in a central way.
        target_size(int): the target length after padding.
        axis(int)：the axis along which to apply padding.
            The default value is -1.
        pad_value(int)：the padding value.
            The default value is 0.0.
    Returns:
        Tensor: the padded tensor. If target_size <= x.shape[axis], the original input tensor is returned
        without padding.
    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        x = F.center_padding(paddle.randn(([8, 10])), target_size=12, axis=1)
        print(x.shape)
        >> [8, 12]

    """
    assert axis < x.ndim, ('axis must be smaller than x.ndim, ' +
                           f'but received aixs={axis},x.ndim={x.ndim}')
    assert x.ndim in [1, 2, 3], (f'only accept 1d/2d/3d tensor, ' +
                                 f'but recieved x.ndim={x.ndim}')

    shape = x.shape
    if target_size <= shape[axis]:
        return x  # nothing to do

    size_diff_hf = (target_size - shape[axis]) // 2
    pad_shape = shape[:]
    pad_shape[axis] = size_diff_hf
    pad_tensor = paddle.ones(pad_shape, x.dtype) * pad_value
    size_diff_hf2 = target_size - shape[axis] - size_diff_hf
    if size_diff_hf2 != size_diff_hf:
        pad_shape = shape[:]
        pad_shape[axis] = size_diff_hf2
        pad_tensor2 = paddle.ones(pad_shape, x.dtype) * pad_value
    else:
        pad_tensor2 = pad_tensor

    return paddle.concat([pad_tensor, x, pad_tensor2], axis=axis)


def stft(x: Tensor,
         n_fft: int = 2048,
         hop_length: Optional[int] = None,
         win_length: Optional[int] = None,
         window: str = 'hann',
         center: bool = True,
         pad_mode: str = 'reflect',
         one_sided: bool = True,
         dtype: str = 'float64'):
    """Compute short-time Fourier transformation(STFT) of a given signal,
    typically an audio waveform.
    The STFT is implemented with strided 1d convolution. The convluational weights are
    not learnable by default. To enable learning, set stop_gradient=False before training.

    Parameters:
        n_fft(int): the number of frequency components of the discrete Fourier transform.
            The default value is 2048.
        hop_length(int|None): the hop length of the short time FFT. If None, it is set to win_length//4.
            The default value is None.
        win_length: the window length of the short time FFt. If None, it is set to same as n_fft.
            The default value is None.
        window(str): the name of the window function applied to the single before the Fourier transform.
            The folllowing window names are supported: 'hamming','hann','kaiser','gaussian',
            'exponential','triang','bohman','blackman','cosine','tukey','taylor'.
            The default value is 'hann'
        center(bool): if True, the signal is padded so that frame t is centered at x[t * hop_length].
            If False, frame t begins at x[t * hop_length]
            The default value is True
        pad_mode(str): the mode to pad the signal if necessary. The supported modes are 'reflect' and 'constant'.
            The default value is 'reflect'.
        one_sided(bool): If True, the output spectrum will have n_fft//2+1 frequency components.
            Otherwise, it will return the full spectrum that have n_fft+1 frequency values.
            The default value is True.
        dtype(str): the datatype used internally for computing fft transform coefficients. 'float64' is
            recommended for higher numerical accuracy.
    Shape:
        - x: 1-D tensor with shape: (signal_length,) or 2-D tensor with shape (N, signal_length).
        - output: 2-D tensor with shape (N, freq_dim, frame_number,2),
        where freq_dim = n_fft+1 if one_sided is False and n_fft//2+1 if True.
        The batch size N is set to 1 if input singal x is 1D tensor.
    Notes:
        This result of stft function is consistent with librosa.stft() for the default value setting.
    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        x = F.istft(paddle.randn(([8, 1025, 32, 2])), signal_length=16000)
        print(x.shape)
        >> [8, 16000]

    """
    assert x.ndim in [
        1, 2
    ], (f'The input signal x must be a 1-d tensor for ' +
        'non-batched signal or 2-d tensor for batched signal, ' +
        f'but received ndim={x.ndim} instead')

    if x.ndim == 1:
        x = x.unsqueeze((0, 1))
    elif x.ndim == 2:
        x = x.unsqueeze(1)

    # By default, use the entire frame.
    if win_length is None:
        win_length = n_fft
    # Set the default hop, if it's not already specified.
    if hop_length is None:
        hop_length = int(win_length // 4)
    fft_window = get_window(window, win_length, fftbins=True, dtype=dtype)
    fft_window = center_padding(fft_window, n_fft)
    dft_mat = dft_matrix(n_fft, dtype=dtype)
    if one_sided:
        out_channels = n_fft // 2 + 1
    else:
        out_channels = n_fft
    weight = fft_window.unsqueeze([1, 2]) * dft_mat[:, 0:out_channels, :]
    weight = weight.transpose([1, 2, 0])
    weight = weight.reshape([-1, weight.shape[-1]]).unsqueeze(1)

    if center:
        x = paddle.nn.functional.pad(x,
                                     pad=[n_fft // 2, n_fft // 2],
                                     mode=pad_mode,
                                     data_format="NCL")
    signal = paddle.nn.functional.conv1d(x,
                                         weight.astype('float32'),
                                         stride=hop_length)

    signal = signal.transpose([0, 2, 1])
    signal = signal.reshape(
        [signal.shape[0], signal.shape[1], signal.shape[2] // 2, 2])
    signal = signal.transpose((0, 2, 1, 3))
    return signal


def istft(x: Tensor,
          n_fft: int = 2048,
          hop_length: Optional[int] = None,
          win_length: Optional[int] = None,
          window: str = 'hann',
          center: bool = True,
          pad_mode: str = 'reflect',
          signal_length: Optional[int] = None,
          dtype: str = 'float64') -> Tensor:
    """Compute inverse short-time Fourier transform(ISTFT) of a given spectrum signal x.
    To accurately recover the input signal, the exact value of parameters should match
    those used in stft.

    Parameters:
        n_fft, hop_length, win_length, window, center, pad_mode: please refer to stft()
        signal_length(int): the origin signal length for exactly aligning recovered signal
            with original signal. If set to None, the length is solely determined by hop_length
            and win_length.
            The default value is None.
        dtype(str): the datatype used internally for computing fft transform coefficients. 'float64' is
            recommended for higher numerical accuracy.
    Shape:
        - x: 1-D tensor with shape: (signal_length,) or 2-D tensor with shape (N, signal_length).
        - output: the signal represented as a 2-D tensor with shape (N, single_length)
            The batch size N is set to 1 if input singal x is 1D tensor.

    Examples:
        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        x = paddle.rand((32, 16000))
        y = F.stft(x, n_fft=512)
        print(x.shape)
        >> [32, 16000]
        z = F.istft(y, n_fft=512, signal_length=16000)
        print(z.shape)
        >> [32, 16000]
        print((z-x).abs().mean())
        >> Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
                [0.00000707])

    """
    assert pad_mode in [
        'constant', 'reflect'
    ], (f'only support "constant" or ' +
        f'"reflect" for pad_mode, but received pad_mode={pad_mode}')

    assert x.ndim in [
        3, 4
    ], (f'The input spectrum x must be a 3-d or 4-d tensor, ' +
        f'but received ndim={x.ndim} instead')

    if x.ndim == 3:
        x = x.unsqueeze(0)

    bs, freq_dim, frame_num, complex_dim = x.shape
    assert freq_dim == n_fft or freq_dim == n_fft // 2 + 1, (
        f'The input spectrum x should have {n_fft} or {n_fft//2+1} frequency ' +
        f'components, but received {freq_dim} instead')

    assert complex_dim == 2, (
        f'The last dimension of input spectrum should be 2 for storing ' +
        f'real and imaginary part of spectrum, but received {complex_dim} instead'
    )

    # By default, use the entire frame.
    if win_length is None:
        win_length = n_fft
    # Set the default hop, if it's not already specified.
    if hop_length is None:
        hop_length = int(win_length // 4)

    assert hop_length < win_length, (
        f'hop_length must be smaller than win_length, ' +
        f'but {hop_length}>={win_length}')

    fft_window = get_window(window, win_length, dtype=dtype)
    fft_window = 1.0 / fft_window
    fft_window = center_padding(fft_window, n_fft)
    fft_window = fft_window.unsqueeze((1, 2))
    idft_mat = fft_window * idft_matrix(n_fft, dtype=dtype) / n_fft
    idft_mat = idft_mat.unsqueeze((0, 1))

    #let's do the inverse transformation
    real = x[:, :, :, 0]
    imag = x[:, :, :, 1]
    if real.shape[1] == n_fft:
        real_full = real
        imag_full = imag
    else:
        real_full = paddle.concat([real, real[:, -2:0:-1]], 1)
        imag_full = paddle.concat([imag, -imag[:, -2:0:-1]], 1)
    part1 = paddle.matmul(idft_mat[:, :, :, :, 0], real_full)
    part2 = paddle.matmul(idft_mat[:, :, :, :, 1], imag_full)
    frames = part1[0] - part2[0]
    signal = deframe(frames, n_fft, hop_length, win_length, signal_length)
    return signal


def spectrogram(x,
                n_fft: int = 2048,
                hop_length: Optional[int] = None,
                win_length: Optional[int] = None,
                window: str = 'hann',
                center: bool = True,
                pad_mode: str = 'reflect',
                power: float = 2.0,
                dtype: str = 'float64') -> Tensor:
    """Compute spectrogram of a given signal, typically an audio waveform.
        The spectorgram is defined as the complex norm of the short-time
        Fourier transformation.

    Parameters:
            n_fft(int): the number of frequency components of the discrete Fourier transform.
                The default value is 2048,
            hop_length(int|None): the hop length of the short time FFT. If None, it is set to win_length//4.
                The default value is None.
            win_length: the window length of the short time FFt. If None, it is set to same as n_fft.
                The default value is None.
            window(str): the name of the window function applied to the single before the Fourier transform.
                The folllowing window names are supported: 'hamming','hann','kaiser','gaussian',
                'exponential','triang','bohman','blackman','cosine','tukey','taylor'.
                The default value is 'hann'
            center(bool): if True, the signal is padded so that frame t is centered at x[t * hop_length].
                If False, frame t begins at x[t * hop_length]
                The default value is True
            pad_mode(str): the mode to pad the signal if necessary. The supported modes are 'reflect'
                and 'constant'.
                The default value is 'reflect'.
            power(float): The power of the complex norm.
                The default value is 2.0
            dtype(str): the datatype used internally for computing fft transform coefficients. 'float64' is
                recommended for higher numerical accuracy.
    Shape:
            - x: 1-D tensor with shape: (signal_length,) or 2-D tensor with shape (N, signal_length).
            - output: 2-D tensor with shape (N, n_fft//2+1, frame_number),
            The batch size N is set to 1 if input singal x is 1D tensor.

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        x = F.spectrogram(paddle.randn((8, 16000,)))
        print(x.shape)
        >> [8, 1025, 32]

      """
    fft_signal = stft(x,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      win_length=win_length,
                      window=window,
                      center=center,
                      pad_mode=pad_mode,
                      one_sided=True,
                      dtype=dtype)
    spectrogram = paddle.square(fft_signal).sum(-1)
    if power == 2.0:
        pass
    else:
        spectrogram = spectrogram**(power / 2.0)
    return spectrogram


def melspectrogram(x: Tensor,
                   sr: int = 22050,
                   n_fft: int = 2048,
                   hop_length: Optional[int] = None,
                   win_length: Optional[int] = None,
                   window: str = 'hann',
                   center: bool = True,
                   pad_mode: str = 'reflect',
                   power: float = 2.0,
                   n_mels: int = 128,
                   f_min: float = 0.0,
                   f_max: Optional[float] = None,
                   htk: bool = True,
                   norm: Union[str, float] = 'slaney',
                   dtype: str = 'float64',
                   to_db: bool = False,
                   **kwargs) -> Tensor:
    """Compute the melspectrogram of a given signal, typically an audio waveform.
        The melspectrogram is also known as filterbank or fbank feature in audio community.
        It is computed by multiplying spectrogram with Mel filter bank matrix.

        Parameters:
            sr(int): the audio sample rate.
                The default value is 22050.
            n_fft(int): the number of frequency components of the discrete Fourier transform.
                The default value is 2048,
            hop_length(int|None): the hop length of the short time FFT. If None, it is set to win_length//4.
                The default value is None.
            win_length: the window length of the short time FFt. If None, it is set to same as n_fft.
                The default value is None.
            window(str): the name of the window function applied to the single before the Fourier transform.
                The folllowing window names are supported: 'hamming','hann','kaiser','gaussian',
                'exponential','triang','bohman','blackman','cosine','tukey','taylor'.
                The default value is 'hann'
            center(bool): if True, the signal is padded so that frame t is centered at x[t * hop_length].
                If False, frame t begins at x[t * hop_length]
                The default value is True
            pad_mode(str): the mode to pad the signal if necessary. The supported modes are 'reflect'
                and 'constant'.
                The default value is 'reflect'.
            power(float): The power of the complex norm.
                The default value is 2.0
            n_mels(int): the mel bins, comman choices are 32, 40, 64, 80, 128.
            f_min(float): the lower cut-off frequency, below which the filter response is zero. Tips:
                set f_min to slightly higher than 0.
                The default value is 0.
            f_max(float): the upper cut-off frequency, above which the filter response is zero.
                If None, it is set to half of the sample rate, i.e., sr//2. Tips: set it a slightly
                smaller than half of sample rate.
                The default value is None.
            htk(bool): whether to use HTK formula in computing fbank matrix.
            norm(str|float): the normalization type in computing fbank matrix.  Slaney-style is used by default.
                You can specify norm=1.0/2.0 to use customized p-norm normalization.
            dtype(str): the datatype of fbank matrix used in the transform. Use float64(default) to increase numerical
                accuracy. Note that the final transform will be conducted in float32 regardless of dtype of fbank matrix.
            to_db(bool): whether to convert the magnitude to db scale.
                The default value is False.
            kwargs: the key-word arguments that are passed to F.power_to_db if to_db is True

        Shape:
            - x: 1-D tensor with shape: (signal_length,) or 2-D tensor with shape (N, signal_length).
            - output: 2-D tensor with shape (N, n_mels, frame_number),
            The batch size N is set to 1 if input singal x is 1D tensor.

        Notes:
            1. The melspectrogram function relies on F.spectrogram and F.compute_fbank_matrix.
            2. The melspectrogram function does not convert magnitude to db by default.

        Examples:

            .. code-block:: python

            import paddle
            import paddleaudio.functional as F
            x = F.melspectrogram(paddle.randn((8, 16000,)))
            print(x.shape)
            >> [8, 128, 32]

    """

    x = spectrogram(x,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=center,
                    pad_mode=pad_mode,
                    power=power,
                    dtype=dtype)
    if f_max is None:
        f_max = sr // 2
    fbank_matrix = compute_fbank_matrix(sr=sr,
                                        n_fft=n_fft,
                                        n_mels=n_mels,
                                        f_min=f_min,
                                        f_max=f_max,
                                        htk=htk,
                                        norm=norm,
                                        dtype=dtype)
    fbank_matrix = fbank_matrix.unsqueeze(0)
    mel_feature = paddle.matmul(fbank_matrix, x.astype(fbank_matrix.dtype))
    if to_db:
        mel_feature = power_to_db(mel_feature, **kwargs)

    return mel_feature


def mfcc(x,
         sr: int = 22050,
         spect: Optional[Tensor] = None,
         n_mfcc: int = 20,
         dct_norm: str = 'ortho',
         lifter: int = 0,
         dtype: str = 'float64',
         **kwargs) -> Tensor:
    """Compute Mel-frequency cepstral coefficients (MFCCs) give an input waveform.

     Parameters:
            sr(int): the audio sample rate.
                The default value is 22050.
            spect(None|Tensor): the melspectrogram tranform result(in db scale). If None, the melspectrogram will be
                computed using `MelSpectrogram` functional and further converted to db scale using `F.power_to_db`
                The default value is None.
            n_mfcc(int): the number of coefficients.
                The default value is 20.
            dct_norm: the normalization type of dct matrix. See `dct_matrix` for more details.
                The default value is 'ortho'.
            lifter(int): if lifter > 0, apply liftering(cepstral filtering) to the MFCCs.
                If lifter = 0, no liftering is applied.
                Setting lifter >= 2 * n_mfcc emphasizes the higher-order coefficients.
                As lifter increases, the coefficient weighting becomes approximately linear.
                The default value is 0.
            dtype(str): the datatype used internally in computing MFCC.


    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.functional as F
        x = paddle.randn((8, 16000))  # the waveform
        y = F.mfcc(x,
                sr=16000,
                n_mfcc=20,
                n_mels=64,
                n_fft=512,
                win_length=512,
                hop_length=160)

        print(y.shape)
        >> [8, 20, 101]
    """

    if spect is None:
        spect = melspectrogram(x, sr=sr, dtype=dtype,
                               **kwargs)  #[batch,n_mels,frames]
        spect = power_to_db(spect)  # default top_db is 80

    n_mels = spect.shape[1]
    if n_mfcc > n_mels:
        raise ValueError('Value of n_mfcc cannot be larger than n_mels')

    M = dct_matrix(n_mfcc, n_mels, dct_norm=dct_norm, dtype=dtype)
    out = M.transpose([1, 0]).unsqueeze_(0) @ spect
    if lifter > 0:
        factor = paddle.sin(math.pi *
                            paddle.arange(1, 1 + n_mfcc, dtype=dtype) / lifter)
        return out @ factor.unsqueeze([0, 2])
    elif lifter == 0:
        return out
    else:
        raise ValueError(f"MFCC lifter={lifter} must be a non-negative number")
