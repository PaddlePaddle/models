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

mfcc_test_data = [
    (512, 80, 40, True, 7600, 'gpu'),
    (512, 64, 20, False, 8000, 'gpu'),
    (512, 80, 40, True, 8000, 'gpu'),
    (512, 64, 20, False, 8000, 'gpu'),
    (512, 64, 20, False, 8000, 'gpu'),
    (512, 40, 20, True, 8000, 'gpu'),
    (512, 80, 40, True, 7600, 'cpu'),
    (512, 64, 20, False, 8000, 'cpu'),
    (512, 80, 40, True, 8000, 'cpu'),
    (512, 64, 20, False, 8000, 'cpu'),
    (512, 64, 20, False, 8000, 'cpu'),
    (512, 40, 20, True, 8000, 'cpu'),
]


@pytest.mark.parametrize('n_fft,n_mels,n_mfcc,htk,f_max,device', mfcc_test_data)
def test_mfcc_case(n_fft, n_mels, n_mfcc, htk, f_max, device):
    paddle.set_device(device)
    hop_length = 160
    win_length = 512
    window = 'hann'
    pad_mode = 'constant'
    power = 2.0
    sample_rate = 16000
    center = True
    f_min = 0.0
    signal, _ = load_example_audio1()
    signal_tensor = paddle.to_tensor(signal)
    # for librosa, the norm is default to 'slaney'
    expected = librosa.feature.mfcc(signal,
                                    sr=sample_rate,
                                    n_mfcc=n_mfcc,
                                    n_fft=win_length,
                                    hop_length=hop_length,
                                    win_length=win_length,
                                    window=window,
                                    center=center,
                                    n_mels=n_mels,
                                    pad_mode=pad_mode,
                                    fmin=f_min,
                                    fmax=f_max,
                                    htk=htk,
                                    power=2.0)

    mfcc_transform = paddleaudio.transforms.MFCC(sr=sample_rate,
                                                 n_mfcc=n_mfcc,
                                                 n_fft=win_length,
                                                 hop_length=hop_length,
                                                 win_length=win_length,
                                                 window=window,
                                                 center=center,
                                                 n_mels=n_mels,
                                                 pad_mode=pad_mode,
                                                 f_min=f_min,
                                                 f_max=f_max,
                                                 htk=htk,
                                                 norm='slaney')
    paddle_mfcc = mfcc_transform(signal_tensor)

    paddle_librosa_diff = np.mean(np.abs(expected - paddle_mfcc.numpy()))
    assert paddle_librosa_diff < 2.5e-5

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
            'f_min': f_min,
            'f_max': f_max,
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
