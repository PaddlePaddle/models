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

from typing import Optional

import paddle
import paddle.nn as nn
import paddleaudio.functional as F
from paddle import Tensor

__all__ = [
    'STFT', 'ISTFT', 'Spectrogram', 'MelSpectrogram', 'LogMelSpectrogram',
    'RandomMasking', 'CenterPadding', 'RandomCropping', 'Compose'
]


class STFT(nn.Layer):
    """Compute short-time Fourier transformation(STFT) of a given signal, typically an audio waveform.
    The STFT is implemented with strided nn.Conv1D, and the weight is not learnable by default. To fine-tune the Fourier
    coefficients, set stop_gradient=False before training.

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
        pad_mode(str): the mode to pad the signal if necessary. The supported modes are 'reflect' and 'constant'.
        The default value is 'reflect'.

        one_sided(bool): If True, the output spectrum will have n_fft//2+1 frequency components.
            Otherwise, it will return the full spectrum that have n_fft+1 frequency values.
            The default value is True.
    Shape:
        - x: 1-D tensor with shape: (signal_length,) or 2-D tensor with shape (batch, signal_length).
        - output: 2-D tensor with shape [batch_size, freq_dim, frame_number,2], where freq_dim = n_fft+1 if one_sided is False and n_fft//2+1 if True.
        The batch_size is set to 1 if input singal x is 1D tensor.
    Notes:
        This result of stft transform is consistent with librosa.stft() in the default value setting.
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
        assert pad_mode in ['constant', 'reflect']
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
        fft_window = F.pad_center(fft_window, n_fft)
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
        ], f'The input signal x must be a 1-d tensor for non-batched signal or 2-d tensor for batched signal,but received ndim={input.ndim} instead'
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
        return  self.__class__.__name__ +f'(n_fft={self.n_fft}, hop_length={self.hop_length}, '\
               f'win_length={self.win_length}, window="{self.window}")'


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
        The spectorgram is defined as the complex norm of the short-time Fourier transformation.

        Notes:
            The spectrogram transform relies on STFT transform to compute the spectrogram. By default,
             the weight is not learnable. To fine-tune the Fourier coefficients, set stop_gradient=False before training.
        """
        super(Spectrogram, self).__init__()

        self.power = power
        self._stft = STFT(n_fft, hop_length, win_length, window, center,
                          pad_mode)

    def __repr__(self, ):
        p_repr = str(self._stft).split('(')[-1].split(')')[0]
        local_repr = f'power={self.power}'
        return self.__class__.__name__ + '(' + p_repr + ', ' + local_repr + ')'

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
                 fmin: float = 0.0,
                 fmax: Optional[float] = None):
        """Compute the melspectrogram of a given signal, typically an audio waveform.
        THe melspectrogram is also known as filterbank or fbank feature in audio community.

        Notes:
            The melspectrogram transform relies on Spectrogram transform and paddleaudio.functional.compute_fbank_matrix. By default,
            the Fourier coefficients are not learnable. To fine-tune the Fourier coefficients, set stop_gradient=False before training.
            The fbank matrix is handcrafted and not learnable even if stop_gradient=False.
        """
        super(MelSpectrogram, self).__init__()

        self._spectrogram = Spectrogram(n_fft, hop_length, win_length, window,
                                        center, pad_mode, power)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        if fmax is None:
            fmax = sr // 2
        self.fbank_matrix = F.compute_fbank_matrix(sr=sr,
                                                   n_fft=n_fft,
                                                   n_mels=n_mels,
                                                   fmin=fmin,
                                                   fmax=fmax)
        self.fbank_matrix = self.fbank_matrix.unsqueeze(0)
        self.register_buffer('fbank_matrix', self.fbank_matrix)

    def forward(self, x: Tensor) -> Tensor:
        spect_feature = self._spectrogram(x)
        mel_feature = paddle.matmul(self.fbank_matrix, spect_feature)
        return mel_feature

    def __repr__(self):

        p_repr = str(self._spectrogram).split('(')[-1].split(')')[0]
        local_repr = f'n_mels={self.n_mels}, fmin={self.fmin}, fmax={self.fmax}'
        return self.__class__.__name__ + '(' + local_repr + ', ' + p_repr + ')'


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
                 fmin: float = 0.0,
                 fmax: Optional[float] = None):
        """Compute log-mel-spectrogram (also known as LogFBank) feature of a given signal, typically an audio waveform.



        Notes:
            The LogMelSpectrogram transform relies on MelSpectrogram transform to compute spectrogram in mel-scale,
             and then use paddleaudio.functional.power_to_db to convert it into log-scale, also known as decibel(dB) scale.
             By default, the weight is not learnable. To fine-tune the Fourier coefficients, set stop_gradient=False before training.
        """
        super(LogMelSpectrogram, self).__init__()
        self._melspectrogram = MelSpectrogram(sr, n_fft, hop_length, win_length,
                                              window, center, pad_mode, power,
                                              n_mels, fmin, fmax)

    def forward(self, x: Tensor) -> Tensor:
        mel_feature = self._melspectrogram(x)
        log_mel_feature = F.power_to_db(mel_feature)
        return log_mel_feature

    def __repr__(self):
        p_repr = str(self._melspectrogram)
        return self.__class__.__name__ + '(' + p_repr.split('(')[-1].split(
            ')')[0] + ')'


class ISTFT(nn.Layer):
    """Compute inverse short-time Fourier transform(ISTFT) of a given spectrum signal.

    Notes
        This ISTFT is the inverse of STFT.
    """
    def __init__(self,
                 n_fft: int = 2048,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: str = 'hann',
                 center: bool = True,
                 pad_mode: str = 'reflect'):
        super(ISTFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']
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

        assert self.hop_length < self.win_length, f'hop_length must be smaller than win_length, but {self.hop_length}>={self.win_length}'

        fft_window = F.get_window(window, self.win_length)
        fft_window = 1.0 / fft_window
        fft_window = F.pad_center(fft_window, n_fft)
        fft_window = fft_window.unsqueeze((1, 2))
        self.idft_mat = fft_window * F.idft_matrix(n_fft) / n_fft
        self.idft_mat = self.idft_mat.unsqueeze((0, 1))

    def forward(self, spectrum: Tensor, signal_length: int) -> Tensor:

        assert spectrum.ndim in [
            3, 4
        ], f'The input spectrum must be a 3-d or 4-d tensor,but received ndim={spectrum.ndim} instead'
        if spectrum.ndim == 3:
            spectrum = spectrum.unsqueeze(0)

        bs, freq_dim, frame_num, complex_dim = spectrum.shape
        assert freq_dim== self.n_fft or freq_dim == self.n_fft//2+1,\
        f'The input spectrum should have {self.n_fft} or {self.n_fft//2+1} frequency components, but received {freq_dim} instead'

        assert complex_dim == 2, \
        f'The last dimension of input spectrum should be 2 for storing real and imaginary part of spectrum, but received {complex_dim} instead'
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
        return self.__class__.__name__ + f'(n_fft={self.n_fft}, hop_length={self.hop_length}, '\
               f'win_length={self.win_length}, window="{self.window}")'


class RandomMasking(nn.Layer):
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
        return self.__class__.__name__ + f'(max_mask_count={self.max_mask_count}, '+\
            f'max_mask_width={self.max_mask_width}, axis={self.axis})'


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
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
    def __init__(self, crop_size: int, axis: int = -1):
        super(RandomCropping, self).__init__()
        self.crop_size = crop_size
        self.axis = axis

    def forward(self, x):
        return F.RandomCropping(x, crop_size=self.crop_size, axis=self.axis)

    def __repr__(self, ):
        return self.__class__.__name__ + f'(crop_size={self.crop_size}, axis={self.axis})'


class CenterPadding(nn.Layer):
    def __init__(self, target_size: int, axis: int = -1):
        super(CenterPadding, self).__init__()
        self.target_size = target_size
        self.axis = axis

    def forward(self, x):
        return F.center_padding(x, self.target_size, axis=self.axis)

    def __repr__(self, ):
        return self.__class__.__name__ + f'(axis={self.axis}, target_size={self.target_size})'


class MuEncoding(nn.Layer):
    def __init__(self, mu: int):
        super(MuEncoding, self).__init__()
        assert mu > 0, f'mu must be postive, but received mu = {mu}'
        self.mu = mu

    def forward(self, x: Tensor) -> Tensor:
        return F.mu_encode(x, mu=self.mu)

    def __repr__(self, ):
        return self.__class__.__name__ + f'(mu={self.mu})'


class MuDecoding(nn.Layer):
    def __init__(self, mu: int):
        super(MuDecoding, self).__init__()
        assert mu > 0, f'mu must be postive, but received mu = {mu}'
        self.mu = mu

    def forward(self, x: Tensor) -> Tensor:
        return F.mu_decode(x, mu=self.mu)

    def __repr__(self, ):
        return self.__class__.__name__ + f'(mu={self.mu})'


class RandomMuCodec(nn.Layer):
    def __init__(self, min_mu: int = 63, max_mu: int = 255):
        super(RandomMuCodec, self).__init__()
        assert min_mu > 0, f'mu must be postive, but received min_mu = {min_mu}'
        assert max_mu > min_mu, f'max_mu must > min_mu, but received max_mu = {max_mu}, min_mu = {min_mu}'
        self.max_mu = max_mu
        self.min_mu = min_mu

    def forward(self, x: Tensor) -> Tensor:
        mu = int(paddle.randint(low=self.min_mu, high=self.max_mu))
        code = F.mu_encode(x, mu=mu)
        x_out = F.mu_decode(code, mu=mu)
        return x_out

    def __repr__(self, ):
        return self.__class__.__name__ + f'(min_mu={self.min_mu}, max_mu={self.max_mu})'
