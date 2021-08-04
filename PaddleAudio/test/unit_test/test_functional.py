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

import numpy as np
import paddle
import paddleaudio
import paddleaudio.functional as F
import pytest
import scipy
import utils

EPS = 1e-8

# def test_hz_mel_convert():
#     hz = np.linspace(0, 32000, 100).astype('float32')
#     mel0 = paddleaudio.utils._librosa.hz_to_mel(hz)
#     mel1 = F.hz_to_mel(paddle.to_tensor(hz)).numpy()
#     hz0 = paddleaudio.utils._librosa.mel_to_hz(mel0)
#     hz1 = F.mel_to_hz(paddle.to_tensor(mel0)).numpy()
#     assert np.allclose(hz0, hz1)
#     assert np.allclose(mel0, mel1)
#     assert np.allclose(hz, hz0)

# def generate_window_test_data():
#     names = [
#         ('hamming', ),
#         ('hann', ),
#         (
#             'taylor',
#             4,
#             30,
#             True,
#         ),
#         #'kaiser',
#         ('gaussian', 100),
#         ('exponential', None, 1.0),
#         ('triang', ),
#         ('bohman', ),
#         ('blackman', ),
#         ('cosine', ),
#     ]
#     win_length = [512, 400, 1024, 2048]
#     fftbins = [True, False]
#     return itertools.product(names, win_length, fftbins)

# @pytest.mark.parametrize('name,win_length,fftbins', generate_window_test_data())
# def test_get_window(name, win_length, fftbins):
#     src = F.get_window(name, win_length, fftbins=fftbins)
#     target = scipy.signal.get_window(name, win_length, fftbins=fftbins)
#     assert np.allclose(src.numpy(), target, atol=1e-5)

p2db_test_data = [
    (1.0, 1e-10, 80),
    (1.0, 1e-10, None),
    (10.0, 1e-3, 60),
    (10.0, 1e-3, None),
]


@pytest.mark.parametrize('ref_value,amin,top_db', p2db_test_data)
def test_power_to_db(ref_value, amin, top_db):
    np_data = np.random.rand(100, 100).astype('float32') + 1e-6
    pd_data = paddle.to_tensor(np_data)
    src = F.power_to_db(pd_data, ref_value, amin, top_db)
    target = paddleaudio.utils._librosa.power_to_db(np_data, ref_value, amin,
                                                    top_db)
    assert np.allclose(src.numpy(), target, atol=1e-5)


# def test_mu_codec():
#     x, _ = utils.load_example_audio1()
#     x = paddle.to_tensor(x)
#     code = F.mu_law_encode(x)
#     xr = F.mu_law_decode(code)
#     assert np.allclose(xr.numpy(), x.numpy(), atol=1e-1)

#     code = F.mu_law_encode(x, mu=1024)
#     xr = F.mu_law_decode(code, mu=1024)
#     assert np.allclose(xr.numpy(), x.numpy(), atol=1e-2)

#     code = F.mu_law_encode(x, mu=65536)
#     xr = F.mu_law_decode(code, mu=65536)
#     assert np.allclose(xr.numpy(), x.numpy(), atol=1e-4)


def test_mel_frequencies():
    src = F.mel_frequencies(n_mels=128, f_min=0.0, f_max=11025.0, htk=False)
    target = paddleaudio.utils._librosa.mel_frequencies(n_mels=128,
                                                        fmin=0.0,
                                                        fmax=11025.0,
                                                        htk=False)
    assert np.allclose(src.numpy(), target)


def test_fft_frequencies():
    src = F.fft_frequencies(16000, 512)
    target = paddleaudio.utils._librosa.fft_frequencies(16000, 512)
    np.allclose(src.numpy(), target)


def test_fbank_matrix():
    src = F.compute_fbank_matrix(sr=16000, n_fft=512, n_mels=128)
    target = paddleaudio.utils._librosa.compute_fbank_matrix(sr=16000,
                                                             n_fft=512,
                                                             n_mels=128)
    assert np.allclose(src.numpy(), target, atol=1e-7)  # cannot reach 1e-8
