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
from utils import load_example_audio1


def generate_test():
    sr = [16000]
    n_fft = [512, 1024]
    hop_length = [160, 400]
    win_length = [512]
    window = ['hann', 'hamming', ('gaussian', 50)]
    center = [True, False]
    pad_mode = ['reflect', 'constant']
    dtype = ['float32', 'float64']
    device = ['gpu', 'cpu']

    args = [
        sr, n_fft, hop_length, win_length, window, center, pad_mode, dtype,
        device
    ]
    return itertools.product(*args)


@pytest.mark.parametrize(
    'sr,n_fft,hop_length,win_length,window,center,pad_mode,dtype,device',
    generate_test())
def test_case(sr, n_fft, hop_length, win_length, window, center, pad_mode,
              dtype, device):

    if dtype == 'float32':
        if n_fft < 1024:
            max_err = 5e-6
        else:
            max_err = 7e-6
        min_err = 1e-8
    else:  #float64
        max_err = 6.0e-08
        min_err = 1e-10

    paddle.set_device(device)

    signal, _ = load_example_audio1()
    signal_tensor = paddle.to_tensor(signal)  #.to(device)

    stft = paddleaudio.transforms.STFT(n_fft=n_fft,
                                       hop_length=hop_length,
                                       win_length=win_length,
                                       window=window,
                                       center=center,
                                       pad_mode=pad_mode,
                                       dtype=dtype)

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

    assert err <= max_err
    assert err >= min_err
