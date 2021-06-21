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
from paddleaudio.features.core import melspectrogram
import numpy as np
import paddle
import paddleaudio
from paddleaudio.transforms import STFT, MelSpectrogram
import pytest
import librosa
EPS = 1e-8
import itertools

# test case for stft
def generate_stft_test():
    n_fft=[512,1024,2048]
    hop_length = [160,400]
    win_length = [512,320]
    pad_mode = ['reflect','constant']
    args = [n_fft,hop_length,win_length,pad_mode]
    return itertools.product(*args)

@pytest.mark.parametrize('n_fft,hop_length,win_length,pad_mode', generate_stft_test())
def test_stft(n_fft, hop_length, win_length, pad_mode):
    sample_rate = 16000
    data_length = sample_rate * 5
    window = 'hann'
    center = True
    np_data = np.random.uniform(-1, 1, data_length).astype('float32')
    pt_data = paddle.to_tensor(np_data)  #.to(device)

    stft = STFT(n_fft=n_fft,
                  hop_length=hop_length,
                  win_length=win_length,
                  window=window,
                  center=center,
                  pad_mode=pad_mode)

    src = stft(pt_data.unsqueeze(0)).numpy()[0]

    target = paddleaudio.features.stft(np_data,
                                       n_fft=n_fft,
                                       win_length=win_length,
                                       hop_length=hop_length,
                                       window=window,
                                       center=center,
                                       pad_mode=pad_mode)

    assert np.allclose(target.real,src[:, :, 0],rtol=1e-5,atol=1e-2)
    assert np.allclose(target.imag,src[:, :, 1],rtol=1e-5,atol=1e-2)



# test case for stft
def generate_mel_test():
    sr=[16000]
    n_fft=[512,1024]
    hop_length = [160,400]
    win_length = [512]
    window=['hann','hamming',('gaussian',50)]
    center=[True,False]
    pad_mode = ['reflect','constant']
    power = [1.0,2.0]
    n_mels = [120,32]
    fmin = [0,10]
    fmax = [8000,None]
    args = [sr,n_fft,hop_length,win_length,window,center,pad_mode,power,n_mels,fmin,fmax]
    return itertools.product(*args)

@pytest.mark.parametrize('sr,n_fft,hop_length,win_length,window,center,pad_mode,power,n_mels,fmin,fmax', generate_mel_test())
def test_melspectrogram(sr,n_fft, hop_length, win_length,window,center, pad_mode,power,
                n_mels,fmin,fmax):

    melspectrogram = MelSpectrogram(sr,n_fft, hop_length, win_length,window,center, pad_mode,power,
                n_mels,fmin,fmax)
    data_length = 32000 * 5
    np_data = np.random.uniform(-1, 1, data_length).astype('float32')
    pt_data = paddle.to_tensor(np_data)  #.to(device)

    src = melspectrogram(pt_data.unsqueeze(0))

    target = librosa.feature.melspectrogram(np_data,
                                        sr = sr,
                                        n_fft=n_fft,
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    window=window,
                                    center=center,
                                    pad_mode=pad_mode,power = power,
                                        n_mels = n_mels, fmin = fmin, fmax = fmax)

    assert np.allclose(src.numpy()[0],target,atol=1e-3)



if __name__ == '__main__':
    args = list(generate_mel_test())[0]
    test_melspectrogram(*args)
