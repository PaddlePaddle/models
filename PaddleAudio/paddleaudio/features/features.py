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

import librosa
import numpy as np
import paddle

__all__ = ['mel_spect', 'linear_spect', 'log_spect']


#mel
def mel_spect(y,
              sample_rate=16000,
              window_size=512,
              hop_length=320,
              mel_bins=64,
              fmin=50,
              fmax=14000,
              window='hann',
              center=True,
              pad_mode='reflect',
              ref=1.0,
              amin=1e-10,
              top_db=None):
    """ compute mel-spectrogram from input waveform y.
    Create a Mel filter-bank.
    This produces a linear transformation matrix to project
    FFT bins onto Mel-frequency bins.

    """

    s = librosa.stft(y,
                     n_fft=window_size,
                     hop_length=hop_length,
                     win_length=window_size,
                     window=window,
                     center=center,
                     pad_mode=pad_mode)

    power = np.abs(s)**2
    melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax)
    mel = np.matmul(melW, power)
    db = librosa.power_to_db(mel, ref=ref, amin=amin, top_db=None)
    return db


def linear_spect(y,
                 sample_rate=16000,
                 window_size=512,
                 hop_length=320,
                 window='hann',
                 center=True,
                 pad_mode='reflect',
                 power=2):

    s = librosa.stft(y,
                     n_fft=window_size,
                     hop_length=hop_length,
                     win_length=window_size,
                     window=window,
                     center=center,
                     pad_mode=pad_mode)

    return np.abs(s)**power


def log_spect(y,
              sample_rate=16000,
              window_size=512,
              hop_length=320,
              window='hann',
              center=True,
              pad_mode='reflect',
              power=2.0,
              offset=1.0):

    s = librosa.stft(
        y,
        n_fft=window_size,
        hop_length=hop_length,
        win_length=window_size,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    s = np.abs(s)**power

    return np.log(offset + s)  # remove
