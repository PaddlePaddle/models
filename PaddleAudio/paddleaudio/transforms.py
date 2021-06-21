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


class STFT(nn.Layer):
    """Compute short-time Fourier transform(STFT) of a given signal, typically an audio waveform.
    The STFT is implemented with strided nn.Conv1D, and the weight is not learnable by default. To fine-tune the Fourier
    coefficients, set stop_gradient=False before training.

    Notes
        This transform is consistent of librosa.stft.
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

    def forward(self, input: Tensor):

        x = input.unsqueeze(1)
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
        return f'stft(n_fft:{self.n_fft}, hop_length:{self.hop_length}, '\
               f'win_length:{self.win_length}, window:{self.window})'


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

        Notes:
            The spectrogram transform relies on STFT transform to compute the spectrogram. By default,
             the weight is not learnable. To fine-tune the Fourier coefficients, set stop_gradient=False before training.
        """
        super(Spectrogram, self).__init__()

        self.power = power
        self._stft = STFT(n_fft, hop_length, win_length, window, center,
                          pad_mode)

    def __repr__(self, ):
        return f'Spectrogram(n_fft:{self.n_fft}, hop_length:{self.hop_length}, '\
               f'win_length:{self.win_length}, window:{self.window}, power:{self.power})'

    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 2, f'input must satisfy input.ndim==2, but received input.dim = {input.ndim}'
        # print(type(super()))
        fft_signal = self._stft(input)
        # (batch_size, n_fft // 2 + 1, time_steps, 2)
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
                 n_mels: int = 64,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None):
        """Compute spectrogram of a given signal, typically an audio waveform.

        Notes:
            The mel-spectrogram transform relies on Spectrogram transform and paddleaudio.functional.compute_fbank_matrix. By default,
             the weight is not learnable. To fine-tune the Fourier coefficients, set stop_gradient=False before training.
        """
        super(MelSpectrogram, self).__init__()

        self._spectrogram = Spectrogram(n_fft, hop_length, win_length, window,
                                        center, pad_mode, power)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        if fmax == None:
            fmax = sr // 2
        self.fbank_matrix = F.compute_fbank_matrix(sr=sr,
                                                   n_fft=n_fft,
                                                   n_mels=n_mels,
                                                   fmin=fmin,
                                                   fmax=fmax)
        self.fbank_matrix = self.fbank_matrix.unsqueeze(0)

    def forward(self, input: Tensor) -> Tensor:
        spectrogram = self._spectrogram(input)
        mel_spectrogram = paddle.bmm(self.fbank_matrix, spectrogram)
        return mel_spectrogram

    def __repr__(self):
        return f'MelSpectrogram(n_mels:{self.n_mels}, fmin:{self.fmin}, fmax:{self.fmax}, '\
                f'n_fft:{self.n_fft}, hop_length:{self.hop_length}, '\
               f'win_length:{self.win_length}, power:{self.power})'


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

    def forward(self, input: Tensor) -> Tensor:
        mel_spectrogram = self._melspectrogram(input)
        log_mel_spectrogram = F.power_to_db(mel_spectrogram)
        return log_mel_spectrogram

    def __repr__(self):
        return f'LogMelSpectrogram(n_mels:{self.n_mels}, fmin:{self.fmin}, fmax:{self.fmax}, '\
                f'n_fft:{self.n_fft}, hop_length:{self.hop_length}, '\
               f'win_length:{self.win_length}, power:{self.power})'


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

        fft_window = F.get_window(window, win_length)
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
        r = part1[0] - part2[0]
        valid = r[:, self.n_fft // 2 - self.win_length // 2:self.n_fft // 2 +
                  self.win_length // 2, :]
        overlap = (self.win_length - self.hop_length) // 2
        signal = paddle.zeros((
            bs,
            self.hop_length * frame_num,
        ))
        for i in range(frame_num):
            pos = i * self.hop_length
            signal[:, pos:pos +
                   self.hop_length] = valid[:,
                                            overlap:self.win_length - overlap,
                                            i]
        diff = signal.shape[-1] - signal_length
        reconstructed = signal[:, diff // 2:-diff // 2]
        return reconstructed

    def __repr__(self, ):
        return f'Inverse stft(n_fft:{self.n_fft}, hop_length:{self.hop_length}, '\
               f'win_length:{self.win_length}, window:{self.window})'
