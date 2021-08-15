# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import itertools

import librosa
import numpy as np
import paddle
import paddleaudio
import pytest


def generate_mel_test():
    sr = [16000]
    n_fft = [512, 1024]
    hop_length = [160, 400]
    win_length = [512]
    window = ['hann', 'hamming', ('gaussian', 50)]
    center = [True, False]
    pad_mode = ['reflect', 'constant']
    power = [1.0, 2.0]
    n_mels = [80, 64, 32]
    fmin = [0, 10]
    fmax = [8000, None]
    dtype = ['float32', 'float64']
    device = ['gpu', 'cpu']
    args = [
        sr, n_fft, hop_length, win_length, window, center, pad_mode, power,
        n_mels, fmin, fmax, dtype, device
    ]
    return itertools.product(*args)


@pytest.mark.parametrize(
    'sr,n_fft,hop_length,win_length,window,center,pad_mode,power,n_mels,f_min,f_max,dtype,device',
    generate_mel_test())
def test_case(sr, n_fft, hop_length, win_length, window, center, pad_mode,
              power, n_mels, f_min, f_max, dtype, device):

    paddle.set_device(device)
    signal, sr = paddleaudio.load('./test/unit_test/test_audio.wav')
    signal_tensor = paddle.to_tensor(signal)
    paddle_cpu_feat = paddleaudio.functional.melspectrogram(
        signal_tensor,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        n_mels=n_mels,
        pad_mode=pad_mode,
        f_min=f_min,
        f_max=f_max,
        htk=True,
        norm='slaney',
        dtype=dtype)

    librosa_feat = librosa.feature.melspectrogram(signal,
                                                  sr=16000,
                                                  n_fft=n_fft,
                                                  hop_length=hop_length,
                                                  win_length=win_length,
                                                  window=window,
                                                  center=center,
                                                  n_mels=n_mels,
                                                  pad_mode=pad_mode,
                                                  power=2.0,
                                                  norm='slaney',
                                                  htk=True,
                                                  fmin=f_min,
                                                  fmax=f_max)
    err = np.mean(np.abs(librosa_feat - paddle_cpu_feat.numpy()))
    if dtype == 'float64':
        assert err < 1.0e-07
    else:
        assert err < 5.0e-07
