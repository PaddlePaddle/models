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
import librosa
import numpy as np
import paddle
import paddleaudio
import pytest
from utils import load_example_audio1


def test_stft_case1():

    paddle.set_device('cpu')

    n_fft = 512
    hop_length = 160
    win_length = 400
    window = 'hann'
    pad_mode = 'constant'
    sample_rate = 16000
    center = True
    signal, _ = load_example_audio1()
    signal_tensor = paddle.to_tensor(signal)  #.to(device)

    stft = paddleaudio.transforms.STFT(n_fft=n_fft,
                                       hop_length=hop_length,
                                       win_length=win_length,
                                       window=window,
                                       center=center,
                                       pad_mode=pad_mode)

    paddle_feat = stft(signal_tensor.unsqueeze(0))[0]

    target = paddleaudio.utils._librosa.stft(signal,
                                             n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             window=window,
                                             center=center,
                                             pad_mode=pad_mode)
    librosa_feat = np.concatenate(
        [target.real[..., None], target.imag[..., None]], -1)
    err = np.mean(np.abs(librosa_feat - paddle_feat.numpy()))
    assert err < 1.0e-07


def test_stft_case2():

    paddle.set_device('cpu')

    n_fft = 1024
    hop_length = 160
    win_length = 1024
    window = 'hann'
    pad_mode = 'constant'
    sample_rate = 16000
    center = True
    signal, _ = load_example_audio1()
    signal_tensor = paddle.to_tensor(signal)
    stft = paddleaudio.transforms.STFT(n_fft=n_fft,
                                       hop_length=hop_length,
                                       win_length=win_length,
                                       window=window,
                                       center=center,
                                       pad_mode=pad_mode)

    paddle_feat = stft(signal_tensor.unsqueeze(0))[0]

    target = paddleaudio.utils._librosa.stft(signal,
                                             n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             window=window,
                                             center=center,
                                             pad_mode=pad_mode)
    librosa_feat = np.concatenate(
        [target.real[..., None], target.imag[..., None]], -1)
    err = np.mean(np.abs(librosa_feat - paddle_feat.numpy()))
    assert err < 1.0e-07


def test_stft_case3():

    paddle.set_device('gpu')

    n_fft = 512
    hop_length = 160
    win_length = 400
    window = 'hann'
    pad_mode = 'constant'
    sample_rate = 16000
    center = True
    signal, _ = load_example_audio1()
    signal_tensor = paddle.to_tensor(signal)  #.to(device)

    stft = paddleaudio.transforms.STFT(n_fft=n_fft,
                                       hop_length=hop_length,
                                       win_length=win_length,
                                       window=window,
                                       center=center,
                                       pad_mode=pad_mode)

    paddle_feat = stft(signal_tensor.unsqueeze(0))[0]

    target = paddleaudio.utils._librosa.stft(signal,
                                             n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             window=window,
                                             center=center,
                                             pad_mode=pad_mode)
    librosa_feat = np.concatenate(
        [target.real[..., None], target.imag[..., None]], -1)
    err = np.mean(np.abs(librosa_feat - paddle_feat.numpy()))
    assert err < 1.0e-07


def test_stft_case4():

    paddle.set_device('gpu')

    n_fft = 1024
    hop_length = 160
    win_length = 1024
    window = 'hann'
    pad_mode = 'constant'
    sample_rate = 16000
    center = True
    signal, _ = load_example_audio1()
    signal_tensor = paddle.to_tensor(signal)
    stft = paddleaudio.transforms.STFT(n_fft=n_fft,
                                       hop_length=hop_length,
                                       win_length=win_length,
                                       window=window,
                                       center=center,
                                       pad_mode=pad_mode)

    paddle_feat = stft(signal_tensor.unsqueeze(0))[0]

    target = paddleaudio.utils._librosa.stft(signal,
                                             n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             window=window,
                                             center=center,
                                             pad_mode=pad_mode)
    librosa_feat = np.concatenate(
        [target.real[..., None], target.imag[..., None]], -1)
    err = np.mean(np.abs(librosa_feat - paddle_feat.numpy()))
    assert err < 1.0e-06
