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
from paddleaudio.transforms import ISTFT, STFT, MelSpectrogram
from paddleaudio.utils._librosa import melspectrogram

paddle.set_device('cpu')
EPS = 1e-8
import itertools

from utils import load_example_audio1


# test case for stft
def generate_stft_test():
    n_fft = [512, 1024]
    hop_length = [160, 320]
    window = [
        'hann',
        'hamming',
        ('gaussian', 100),  #, ('tukey', 0.5),
        'blackman'
    ]  #'bohman'
    win_length = [500, 400]
    pad_mode = ['reflect', 'constant']
    args = [n_fft, hop_length, window, win_length, pad_mode]
    return itertools.product(*args)


# @pytest.mark.parametrize('n_fft,hop_length,window,win_length,pad_mode',
#                          generate_stft_test())
# def test_istft(n_fft, hop_length, window, win_length, pad_mode):
#     sample_rate = 16000
#     signal_length = sample_rate * 5
#     center = True
#     signal = np.random.uniform(-1, 1, signal_length).astype('float32')
#     signal_tensor = paddle.to_tensor(signal)  #.to(device)

#     stft = STFT(n_fft=n_fft,
#                 hop_length=hop_length,
#                 win_length=win_length,
#                 window=window,
#                 center=center,
#                 pad_mode=pad_mode)

#     spectrum = stft(signal_tensor.unsqueeze(0))

#     istft = ISTFT(n_fft=n_fft,
#                   hop_length=hop_length,
#                   win_length=win_length,
#                   window=window,
#                   center=center,
#                   pad_mode=pad_mode)

#     reconstructed = istft(spectrum, signal_length)
#     assert np.allclose(signal, reconstructed[0].numpy(), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize('n_fft,hop_length,window,win_length,pad_mode',
                         generate_stft_test())
def test_stft(n_fft, hop_length, window, win_length, pad_mode):
    sample_rate = 16000
    signal_length = sample_rate * 5
    center = True
    #signal = paddleaudio.load('./test_audio.wav')
    signal, _ = load_example_audio1()
    signal = np.random.uniform(-1, 1, signal_length).astype('float32')
    signal_tensor = paddle.to_tensor(signal)  #.to(device)

    stft = STFT(n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode)

    src = stft(signal_tensor.unsqueeze(0)).numpy()[0]

    target = paddleaudio.utils._librosa.stft(signal,
                                             n_fft=n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length,
                                             window=window,
                                             center=center,
                                             pad_mode=pad_mode)

    tol = 1e-4
    assert np.allclose(target.real, src[:, :, 0], rtol=tol, atol=tol)
    assert np.allclose(target.imag, src[:, :, 1], rtol=tol, atol=tol)


# def generate_mel_test():
#     sr = [16000]
#     n_fft = [512, 1024]
#     hop_length = [160, 400]
#     win_length = [512]
#     window = ['hann', 'hamming', ('gaussian', 50)]
#     center = [True, False]
#     pad_mode = ['reflect', 'constant']
#     power = [1.0, 2.0]
#     n_mels = [120, 32]
#     fmin = [0, 10]
#     fmax = [8000, None]
#     args = [
#         sr, n_fft, hop_length, win_length, window, center, pad_mode, power,
#         n_mels, fmin, fmax
#     ]
#     return itertools.product(*args)

# @pytest.mark.parametrize(
#     'sr,n_fft,hop_length,win_length,window,center,pad_mode,power,n_mels,fmin,fmax',
#     generate_mel_test())
# def test_melspectrogram(sr, n_fft, hop_length, win_length, window, center,
#                         pad_mode, power, n_mels, fmin, fmax):

#     melspectrogram = MelSpectrogram(sr, n_fft, hop_length, win_length, window,
#                                     center, pad_mode, power, n_mels, fmin, fmax)
#     signal_length = 32000 * 5
#     signal = np.random.uniform(-1, 1, signal_length).astype('float32')
#     signal_tensor = paddle.to_tensor(signal)  #.to(device)

#     src = melspectrogram(signal_tensor.unsqueeze(0))

#     target = librosa.feature.melspectrogram(signal,
#                                             sr=sr,
#                                             n_fft=n_fft,
#                                             win_length=win_length,
#                                             hop_length=hop_length,
#                                             window=window,
#                                             center=center,
#                                             pad_mode=pad_mode,
#                                             power=power,
#                                             n_mels=n_mels,
#                                             fmin=fmin,
#                                             fmax=fmax)

#     assert np.allclose(src.numpy()[0], target, atol=1e-4)
