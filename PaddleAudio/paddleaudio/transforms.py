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

from typing import Any, List, Optional

import paddle
import paddle.nn as nn
import paddleaudio.functional as F
from paddle import Tensor

__all__ = [
    'STFT',
    'ISTFT',
    'Spectrogram',
    'MelSpectrogram',
    'LogMelSpectrogram',
    'Compose',
    'RandomMasking',
    'CenterPadding',
    'RandomCropping',
    'RandomMuLawCodec',
    'MuLawDecoding',
    'MuLawEncoding',
]


class STFT(nn.Layer):
    """Compute short-time Fourier transformation(STFT) of a given signal,
    typically an audio waveform.

    The STFT is implemented with strided 1d convolution. The convluational weights
    are not learnable by default. To enable learning, set stop_gradient=False before training.

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
        one_sided(bool): If True, the output spectrum will have n_fft//2+1 frequency components.
            Otherwise, it will return the full spectrum that have n_fft+1 frequency values.
            The default value is True.
    Shape:
        - x: 1-D tensor with shape: (signal_length,) or 2-D tensor with shape (batch, signal_length).
        - output: 2-D tensor with shape [batch_size, freq_dim, frame_number,2],
            where freq_dim = n_fft+1 if one_sided is False and n_fft//2+1 if True.
        The batch_size is set to 1 if input singal x is 1D tensor.
    Notes:
        This result of stft transform is consistent with librosa.stft() for the default value setting.

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        stft = T.STFT(n_fft=512)
        x = paddle.randn((8,16000,))
        y = stft(x)
        print(y.shape)
        >> [8, 257, 126, 2]

    """
    def __init__(self,
                 n_fft: int = 2048,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode: str = 'reflect',
                 one_sided: bool = True):

        super(STFT, self).__init__()

        assert pad_mode in [
            'constant', 'reflect'
        ], ('pad_mode must be choosen ' + 'between "constant" and "reflect", ' +
            f'but received pad_mode={pad_mode} instead')

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft
        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)
        fft_window = F.get_window(window, self.win_length, fftbins=True)
        fft_window = F.center_padding(fft_window, n_fft)
        # DFT & IDFT matrix.
        dft_mat = F.dft_matrix(n_fft)
        if one_sided:
            out_channels = n_fft // 2 + 1
        else:
            out_channels = n_fft
        self.conv = nn.Conv1D(1,
                              out_channels * 2,
                              n_fft,
                              stride=self.hop_length,
                              bias_attr=False)
        weight = fft_window.unsqueeze([1, 2]) * dft_mat[:, 0:out_channels, :]
        weight = weight.transpose([1, 2, 0])
        weight = weight.reshape([-1, weight.shape[-1]])
        self.conv.load_dict({'weight': weight.unsqueeze(1)})
        # by default, the STFT is not learnable
        for param in self.parameters():
            param.stop_gradient = True

    def forward(self, x: Tensor):

        assert x.ndim in [
            1, 2
        ], (f'The input signal x must be a 1-d tensor for ' +
            'non-batched signal or 2-d tensor for batched signal, ' +
            f'but received ndim={input.ndim} instead')
        if x.ndim == 1:
            x = x.unsqueeze((0, 1))
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        if self.center:
            x = paddle.nn.functional.pad(x,
                                         pad=[self.n_fft // 2, self.n_fft // 2],
                                         mode=self.pad_mode,
                                         data_format="NCL")
        signal = self.conv(x)
        signal = signal.transpose([0, 2, 1])
        signal = signal.reshape(
            [signal.shape[0], signal.shape[1], signal.shape[2] // 2, 2])
        signal = signal.transpose((0, 2, 1, 3))
        return signal

    def __repr__(self, ):
        return (self.__class__.__name__ +
                f'(n_fft={self.n_fft}, hop_length={self.hop_length}, ' +
                f'win_length={self.win_length}, window="{self.window}")')


class Spectrogram(nn.Layer):
    def __init__(self,
                 n_fft: int = 2048,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode: str = 'reflect',
                 power: float = 2.0):
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

        Notes:
            The Spectrogram transform relies on STFT transform to compute the spectrogram.
            By default, the weights are not learnable. To fine-tune the Fourier coefficients,
            set stop_gradient=False before training.
            For more information, see STFT().

        Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        spectrogram = T.Spectrogram(n_fft=512)
        x = paddle.randn((8,16000,))
        y = spectrogram(x)
        print(y.shape)
        >> [8, 257, 126]


        """
        super(Spectrogram, self).__init__()

        self.power = power
        self._stft = STFT(n_fft, hop_length, win_length, window, center,
                          pad_mode)

    def __repr__(self, ):
        p_repr = str(self._stft).split('(')[-1].split(')')[0]
        l_repr = f'power={self.power}'
        return (self.__class__.__name__ + '(' + p_repr + ', ' + l_repr + ')')

    def forward(self, x: Tensor) -> Tensor:
        fft_signal = self._stft(x)
        spectrogram = paddle.square(fft_signal).sum(-1)
        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram**(self.power / 2.0)
        return spectrogram


class MelSpectrogram(nn.Layer):
    def __init__(self,
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
                 f_max: Optional[float] = None):
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
            n_mels(int): the mel bins.
            f_min(float): the lower cut-off frequency, below which the filter response is zero.
            f_max(float): the upper cut-off frequency, above which the filter response is zeros.


        Notes:
            The melspectrogram transform relies on Spectrogram transform and F.compute_fbank_matrix.
            By default, the Fourier coefficients are not learnable. To fine-tune the Fourier coefficients,
            set stop_gradient=False before training. The fbank matrix is handcrafted and not learnable
            regardless of the setting of stop_gradient.
        Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        melspectrogram = T.MelSpectrogram(n_fft=512,n_mels=64)
        x = paddle.randn((8,16000,))
        y = melspectrogram(x)
        print(y.shape)
        >> [8, 64, 126]

        """
        super(MelSpectrogram, self).__init__()

        self._spectrogram = Spectrogram(n_fft, hop_length, win_length, window,
                                        center, pad_mode, power)
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        if f_max is None:
            f_max = sr // 2
        self.fbank_matrix = F.compute_fbank_matrix(sr=sr,
                                                   n_fft=n_fft,
                                                   n_mels=n_mels,
                                                   f_min=f_min,
                                                   f_max=f_max)
        self.fbank_matrix = self.fbank_matrix.unsqueeze(0)
        self.register_buffer('fbank_matrix', self.fbank_matrix)

    def forward(self, x: Tensor) -> Tensor:
        spect_feature = self._spectrogram(x)
        mel_feature = paddle.matmul(self.fbank_matrix, spect_feature)
        return mel_feature

    def __repr__(self):

        p_repr = str(self._spectrogram).split('(')[-1].split(')')[0]
        l_repr = f'n_mels={self.n_mels}, f_min={self.f_min}, f_max={self.f_max}'
        return (self.__class__.__name__ + '(' + l_repr + ', ' + p_repr + ')')


class LogMelSpectrogram(nn.Layer):
    def __init__(self,
                 sr: int = 22050,
                 n_fft: int = 2048,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode: str = 'reflect',
                 power: float = 2.0,
                 n_mels: int = 64,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None):
        """Compute log-mel-spectrogram(also known as LogFBank) feature of a given signal,
        typically an audio waveform.

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
            n_mels(int): the mel bins.
            f_min(float): the lower cut-off frequency, below which the filter response is zero.
            f_max(float): the upper cut-off frequency, above which the filter response is zeros.

        Notes:
            The LogMelSpectrogram transform relies on MelSpectrogram transform to compute
            spectrogram in mel-scale, and then use paddleaudio.functional.power_to_db to
            convert it into log-scale, also known as decibel(dB) scale.
            By default, the weights are not learnable. To fine-tune the Fourier coefficients,
            set stop_gradient=False before training.

        Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        melspectrogram = T.LogMelSpectrogram(n_fft=512,n_mels=64)
        x = paddle.randn((8,16000,))
        y = melspectrogram(x)
        print(y.shape)
        >> [8, 64, 126]

        """
        super(LogMelSpectrogram, self).__init__()
        self._melspectrogram = MelSpectrogram(sr, n_fft, hop_length, win_length,
                                              window, center, pad_mode, power,
                                              n_mels, f_min, f_max)

    def forward(self, x: Tensor) -> Tensor:
        mel_feature = self._melspectrogram(x)
        log_mel_feature = F.power_to_db(mel_feature)
        return log_mel_feature

    def __repr__(self):
        p_repr = str(self._melspectrogram)
        return self.__class__.__name__ + '(' + p_repr.split('(')[-1].split(
            ')')[0] + ')'


class ISTFT(nn.Layer):
    """Compute inverse short-time Fourier transform(ISTFT) of a given spectrum signal x.
    To accurately recover the input signal, the exact value of parameters should match
    those used in stft.

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

        signal_length(int): the origin signal length for exactly aligning recovered signal
        with original signal. If set to None, the length is solely determined by hop_length
        and win_length.
        The default value is None.
    Shape:
        - x: 1-D tensor with shape: (signal_length,) or 2-D tensor with shape (batch, signal_length).
        - output: the signal represented as a 2-D tensor with shape [batch_size, single_length]
        The batch_size is set to 1 if input singal x is 1D tensor.

     Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        melspectrogram = T.LogMelSpectrogram(n_fft=512,n_mels=64)
        x = paddle.randn((8,16000,))
        y = melspectrogram(x)
        print(y.shape)
        >> [8, 64, 126]

    """
    def __init__(self,
                 n_fft: int = 2048,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode: str = 'reflect'):
        super(ISTFT, self).__init__()

        assert pad_mode in [
            'constant', 'reflect'
        ], ('pad_mode must be chosen ' + 'between "constant" and "reflect", ' +
            f'but received pad_mode={pad_mode} instead')

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft
        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        assert self.hop_length < self.win_length, (
            f'hop_length must be smaller than win_length, ' +
            f'but {self.hop_length}>={self.win_length}')

        fft_window = F.get_window(window, self.win_length)
        fft_window = 1.0 / fft_window
        fft_window = F.center_padding(fft_window, n_fft)
        fft_window = fft_window.unsqueeze((1, 2))
        self.idft_mat = fft_window * F.idft_matrix(n_fft) / n_fft
        self.idft_mat = self.idft_mat.unsqueeze((0, 1))

    def forward(self, spectrum: Tensor, signal_length: int) -> Tensor:

        assert spectrum.ndim == 3 or spectrum.ndim == 4, (
            f'The input spectrum must be a 3-d or 4-d tensor, ' +
            f'but received ndim={spectrum.ndim} instead')

        if spectrum.ndim == 3:
            spectrum = spectrum.unsqueeze(0)

        bs, freq_dim, frame_num, complex_dim = spectrum.shape

        assert freq_dim == self.n_fft or freq_dim == self.n_fft // 2 + 1, (
            f'The input spectrum should have {self.n_fft} ' +
            f'or {self.n_fft//2+1} frequency ' +
            f'components, but received {freq_dim} instead')
        assert complex_dim == 2, (
            f'The last dimension of input spectrum should be 2 for ' +
            f'storing real and imaginary part of spectrum, ' +
            f'but received {complex_dim} instead')
        real = spectrum[:, :, :, 0]
        imag = spectrum[:, :, :, 1]
        if real.shape[1] == self.n_fft:
            real_full = real
            imag_full = imag
        else:
            real_full = paddle.concat([real, real[:, -2:0:-1]], 1)
            imag_full = paddle.concat([imag, -imag[:, -2:0:-1]], 1)
        part1 = paddle.matmul(self.idft_mat[:, :, :, :, 0], real_full)
        part2 = paddle.matmul(self.idft_mat[:, :, :, :, 1], imag_full)
        frames = part1[0] - part2[0]
        signal = F.deframe(frames, self.n_fft, self.hop_length, self.win_length,
                           signal_length)
        return signal

    def __repr__(self, ):
        return self.__class__.__name__ + (
            f'(n_fft={self.n_fft}, hop_length={self.hop_length}, ' +
            f'win_length={self.win_length}, window="{self.window}")')


class RandomMasking(nn.Layer):
    """Apply random masking to the input tensor.
    The input tensor is typically a spectrogram.

    Parameters:
        max_mask_count(int): the maximum number of masking regions.
        max_mask_width(int)：the maximum zero-out width of each region.
        axis(int): the axis along which to apply masking.
            The default value is -1.
    Notes:
        Please refer to paddleaudio.functional.random_masking() for more details.

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        transform = T.RandomMasking(max_mask_count=10,max_mask_width=2,axis=1)
        x = paddle.rand((64,100))
        x = transform(x)
        print((x[0,:]==0).astype('int32').sum())
        >> Tensor(shape=[1], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
            [8])
    """
    def __init__(self,
                 max_mask_count: int = 3,
                 max_mask_width: int = 30,
                 axis: int = -1):
        super(RandomMasking, self).__init__()
        self.axis = axis
        self.max_mask_count = max_mask_count
        self.max_mask_width = max_mask_width

    def forward(self, x):
        return F.random_masking(
            x,
            max_mask_count=self.max_mask_count,
            max_mask_width=self.max_mask_width,
            axis=self.axis,
        )

    def __repr__(self, ):
        return (self.__class__.__name__ +
                f'(max_mask_count={self.max_mask_count},' +
                f'max_mask_width={self.max_mask_width}, axis={self.axis})')


class Compose():
    """Compose a list of transforms and apply them to the input tensor sequentially.

    Parameters:
        transforms: a list of transforms.
    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        x = paddle.randn((2, 18000))
        transform = T.Compose([
            T.RandomCropping(target_size=16000),
            T.MelSpectrogram(sr=16000, n_fft=256, n_mels=64),
            T.RandomMasking()
        ])
        y = transform(x)
        print(y.shape)
        >> [2, 64, 251]

    """
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomCropping(nn.Layer):
    """Apply random cropping to the input tensor.
    The input tensor is typically a spectrogram.

    Parameters:
        target_size(int): the target length after cropping.
        axis(int)：the axis along which to apply cropping.
    Notes:
        Please refer to paddleaudio.functional.RandomCropping() for more details.

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        transform = T.RandomCropping(target_size=8,axis=1)
        y = transform(x)
        print(y.shape)
        >> [64, 8]
        transform = T.RandomCropping(target_size=100,axis=1)
        y = transform(x)
        print(y.shape)
        >> [64, 100]

    """
    def __init__(self, target_size: int, axis: int = -1):
        super(RandomCropping, self).__init__()
        self.target_size = target_size
        self.axis = axis

    def forward(self, x):
        return F.random_cropping(x,
                                 target_size=self.target_size,
                                 axis=self.axis)

    def __repr__(self, ):
        return (self.__class__.__name__ +
                f'(target_size={self.target_size}, axis={self.axis})')


class CenterPadding(nn.Layer):
    """Apply center cropping to the input tensor.

    Parameters:
        target_size(int): the target length after padding.
        axis(int)：the axis along which to apply padding.
    Notes:
        Please refer to paddleaudio.functional.center_padding() for more details.

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        x = paddle.rand((8,10))
        transform = T.CenterPadding(target_size=12,axis=1)
        y = transform(x)
        print(y.shape)
        >> [8, 12]

    """
    def __init__(self, target_size: int, axis: int = -1):
        super(CenterPadding, self).__init__()
        self.target_size = target_size
        self.axis = axis

    def forward(self, x):
        return F.center_padding(x, self.target_size, axis=self.axis)

    def __repr__(self, ):
        return (self.__class__.__name__ +
                f'(axis={self.axis}, target_size={self.target_size})')


class MuLawEncoding(nn.Layer):
    """Apply Mu-law Encoding transform to the input singal, typically an audio waveform.

    Parameters:
        x(Tensor): the input tensor of arbitrary shape to be encoded.
        mu(int): the maximum value (depth) of encoded signal. The signal will be
            clip to be in range [0,mu-1].
            The default value is 256, i.e., 8bit depth.
        quantized(bool): indicate whether the signal will quantized to integers. If True,
            the result will be converted to integer in range [0,mu-1]. Otherwise, the
            resulting signal is in range [-1,1]
    Notes:
        Please refer to paddleaudio.functional.mu_law_encode() for more details.

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        x = paddle.randn((2,8))
        transform = T.MuLawEncoding()
        y = transform(x)
        print(y)
        >> Tensor(shape=[2, 8], dtype=int32, place=CUDAPlace(0), stop_gradient=True,
        [[0  , 252, 77 , 250, 221, 34 , 51 , 0  ],
            [227, 33 , 0  , 255, 11 , 213, 255, 10 ]])
    """
    def __init__(self, mu: int = 256):
        super(MuLawEncoding, self).__init__()
        assert mu > 0, f'mu must be positive, but received mu = {mu}'
        self.mu = mu

    def forward(self, x: Tensor) -> Tensor:
        return F.mu_law_encode(x, mu=self.mu)

    def __repr__(self, ):
        return self.__class__.__name__ + f'(mu={self.mu})'


class MuLawDecoding(nn.Layer):
    """Apply Mu-law decoding to the input tensor, typically an audio waveform.

    Parameters:
        x(Tensor): the input tensor of arbitrary shape to be decoded.
        mu(int): the maximum value (depth) of encoded signal. The signal to be decoded must be
            in range [0,mu-1].
        quantized(bool): indicate whether the signal has been quantized. The value of quantized parameter should be
        consistent with that used in MuLawEncoding.
    Notes:
        Please refer to paddleaudio.functional.mu_law_decode() for more details.
    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        x = paddle.randint(0,255,shape=(2,8))
        transform = T.MuLawDecoding()
        y = transform(x)
        print(y)
        >> Tensor(shape=[2, 8], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            [[-0.01151094, -0.02702747,  0.00796641, -0.91636580,  0.45497340,  0.49667698,  0.01151095, -0.24569811],
                [ 0.21516445, -0.30633399,  0.01291343, -0.01991909, -0.00904676,  0.00105976,  0.03990653, -0.20584014]])

    """
    def __init__(self, mu: int = 256):
        super(MuLawDecoding, self).__init__()
        assert mu > 0, f'mu must be positive, but received mu = {mu}'
        self.mu = mu

    def forward(self, x: Tensor) -> Tensor:
        return F.mu_law_decode(x, mu=self.mu)

    def __repr__(self, ):
        return self.__class__.__name__ + f'(mu={self.mu})'


class RandomMuLawCodec(nn.Layer):
    """Apply Random MuLawEncoding and MuLawDecoding to the input singal.
    This is useful for simulating audio compression and quantization effects and is commonly
    used in training deep neural networks.

    Parameters:
        min_mu(int): the lower bound of mu as a random variable.
        max_mu(int): the upper bound of mu as a random variable. At each time of the transform,
            the exact mu will be randomly chosen from uniform ~ [min_mu, max_mu].
    Notes:
        Please refer to MuLawDecoding() and MuLawEncoding() for more details.

    Examples:

        .. code-block:: python

        import paddle
        import paddleaudio.transforms as T
        x = paddle.randn((2,8))
        transform = T.RandomMuLawCodec()
        y = transform(x)
        print(y)
        >> Tensor(shape=[2, 8], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        [[ 0.61542195, -0.35218054,  0.30605811, -0.12115669, -0.75794631,  0.03876950, -0.23082513, -0.49945647],
            [-0.35218054, -0.87066686, -0.53548712,  1.        , -1.        ,  0.49945661,  1.        , -0.93311179]])

    """
    def __init__(self, min_mu: int = 63, max_mu: int = 255):
        super(RandomMuLawCodec, self).__init__()
        assert min_mu > 0, (f'mu must be positive, ' +
                            f'but received min_mu = {min_mu}')

        assert max_mu > min_mu, (f'max_mu must > min_mu, ' +
                                 f'but received max_mu = {max_mu}, ' +
                                 f'min_mu = {min_mu}')
        self.max_mu = max_mu
        self.min_mu = min_mu

    def forward(self, x: Tensor) -> Tensor:
        mu = int(paddle.randint(low=self.min_mu, high=self.max_mu))
        code = F.mu_law_encode(x, mu=mu)
        x_out = F.mu_law_decode(code, mu=mu)
        return x_out

    def __repr__(self, ):
        return (self.__class__.__name__ +
                f'(min_mu={self.min_mu}, max_mu={self.max_mu})')
