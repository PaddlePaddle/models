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

import numpy as np
import paddle
import paddleaudio
import paddleaudio.functional as F
import pytest

EPS = 1e-8


def test_hz_mel_convert():
    hz = np.linspace(0, 32000, 100).astype('float32')
    mel0 = paddleaudio.core.hz_to_mel(hz)
    mel1 = F.hz_to_mel(paddle.to_tensor(hz)).numpy()
    hz0 = paddleaudio.core.mel_to_hz(mel0)
    hz1 = F.mel_to_hz(paddle.to_tensor(mel0)).numpy()
    assert np.allclose(hz0, hz1)
    assert np.allclose(mel0, mel1)
    assert np.allclose(hz, hz0)


win_test_data = [
    ('hamming', ),
    ('hann', ),
    #'kaiser',
    ('gaussian', 100),
    ('exponential', None, 1.0),
    ('triang', ),
    ('bohman', ),
    ('blackman', ),
    ('cosine', ),
]


@pytest.mark.parametrize('name', win_test_data)
def test_get_window(name):
    assert F.get_window(name, 1024, fftbins=True).shape == [1024]


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
    target = paddleaudio.features.power_to_db(np_data, ref_value, amin, top_db)
    assert np.allclose(src.numpy(), target)

def test_mu_codec():
    x = np.random.rand(16000).astype('float32')
    xt = paddle.to_tensor(x)
    xqt = F.mu_encode(xt) 
    xq = paddleaudio.features.mu_encode(x)
    xqd = paddleaudio.features.mu_decode(xq)
    xqdt = F.mu_decode(xqt) 
    assert np.allclose(xq, xqt.numpy())
    assert np.allclose(xqd, xqdt.numpy())
    