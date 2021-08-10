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

eps_float32 = 1e-3
eps_float64 = 2.2e-5
# Pre-loading to speed up the test
signal, _ = load_example_audio1()
signal_tensor = paddle.to_tensor(signal)


def generate_mfcc_test():
    sr = [16000]
    n_fft = [512]  #, 1024]
    hop_length = [160]  #, 400]
    win_length = [512]
    window = ['hann']  # 'hamming', ('gaussian', 50)]
    center = [True]  #, False]
    pad_mode = ['reflect', 'constant']
    power = [2.0]
    n_mels = [64]  #32]
    fmin = [0, 10]
    fmax = [8000, None]
    dtype = ['float32', 'float64']
    device = ['gpu', 'cpu']
    n_mfcc = [40, 20]
    htk = [True]
    args = [
        sr, n_fft, hop_length, win_length, window, center, pad_mode, power,
        n_mels, fmin, fmax, dtype, device, n_mfcc, htk
    ]
    return itertools.product(*args)


@pytest.mark.parametrize(
    'sr, n_fft, hop_length, win_length, window, center, pad_mode, power,\
        n_mels, fmin, fmax,dtype,device,n_mfcc,htk', generate_mfcc_test())
def test_mfcc_case(sr, n_fft, hop_length, win_length, window, center, pad_mode, power,\
        n_mels, fmin, fmax,dtype,device,n_mfcc,htk):
    # paddle.set_device(device)
    # hop_length = 160
    # win_length = 512
    # window = 'hann'
    # pad_mode = 'constant'
    # power = 2.0
    # sample_rate = 16000
    # center = True
    # f_min = 0.0

    # for librosa, the norm is default to 'slaney'
    expected = librosa.feature.mfcc(signal,
                                    sr=sr,
                                    n_mfcc=n_mfcc,
                                    n_fft=win_length,
                                    hop_length=hop_length,
                                    win_length=win_length,
                                    window=window,
                                    center=center,
                                    n_mels=n_mels,
                                    pad_mode=pad_mode,
                                    fmin=fmin,
                                    fmax=fmax,
                                    htk=htk,
                                    power=2.0)

    paddle_mfcc = paddleaudio.functional.mfcc(signal_tensor,
                                              sr=sr,
                                              n_mfcc=n_mfcc,
                                              n_fft=win_length,
                                              hop_length=hop_length,
                                              win_length=win_length,
                                              window=window,
                                              center=center,
                                              n_mels=n_mels,
                                              pad_mode=pad_mode,
                                              f_min=fmin,
                                              f_max=fmax,
                                              htk=htk,
                                              norm='slaney',
                                              dtype=dtype)

    paddle_librosa_diff = np.mean(np.abs(expected - paddle_mfcc.numpy()))
    if dtype == 'float64':
        assert paddle_librosa_diff < eps_float64
    else:
        assert paddle_librosa_diff < eps_float32

    try:  # if we have torchaudio installed
        import torch
        import torchaudio
        kwargs = {
            'n_fft': win_length,
            'hop_length': hop_length,
            'win_length': win_length,
            # 'window':window,
            'center': center,
            'n_mels': n_mels,
            'pad_mode': pad_mode,
            'f_min': fmin,
            'f_max': fmax,
            'mel_scale': 'htk',
            'norm': 'slaney',
            'power': 2.0
        }
        torch_mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=20,
                                                          log_mels=False,
                                                          melkwargs=kwargs)
        torch_mfcc = torch_mfcc_transform(torch.tensor(signal))
        paddle_torch_mfcc_diff = np.mean(
            np.abs(paddle_mfcc.numpy() - torch_mfcc.numpy()))
        assert paddle_torch_mfcc_diff < 5e-5
        torch_librosa_mfcc_diff = np.mean(np.abs(torch_mfcc.numpy() - expected))
        assert torch_librosa_mfcc_diff < 5e-5
    except:
        pass


#test_mfcc_case(512, 40, 20, True, 8000, 'cpu','float64',eps_float64)
