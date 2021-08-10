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
import paddleaudio as pa
import pytest
from scipy.signal import get_window


def test_data():
    win_length = [256, 512, 1024]
    sym = [True, False]
    device = ['gpu', 'cpu']
    dtype = ['float32', 'float64']
    args = [win_length, sym, device, dtype]
    return itertools.product(*args)


@pytest.mark.parametrize('win_length,sym,device,dtype', test_data())
def test_window(win_length, sym, device, dtype):
    paddle.set_device(device)
    if dtype == 'float64':
        upper_err = 7e-8
        lower_err = 0
    else:
        upper_err = 8e-8
        lower_err = 0

    src = pa.blackman_window(win_length, sym, dtype=dtype).numpy()
    expected = get_window('blackman', win_length, not sym)
    assert np.mean(np.abs(src - expected)) < upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err

    src = pa.bohman_window(win_length, sym, dtype=dtype).numpy()
    expected = get_window('bohman', win_length, not sym)
    assert np.mean(np.abs(src - expected)) < upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err

    src = pa.triang_window(win_length, sym, dtype=dtype).numpy()
    expected = get_window('triang', win_length, not sym)
    assert np.mean(np.abs(src - expected)) < upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err

    src = pa.hamming_window(win_length, sym, dtype=dtype).numpy()
    expected = get_window('hamming', win_length, not sym)
    assert np.mean(np.abs(src - expected)) < upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err

    src = pa.hann_window(win_length, sym, dtype=dtype).numpy()
    expected = get_window('hann', win_length, not sym)
    assert np.mean(np.abs(src - expected)) < upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err

    src = pa.tukey_window(win_length, 0.5, sym, dtype=dtype).numpy()
    expected = get_window(('tukey', 0.5), win_length, not sym)
    assert np.mean(np.abs(src - expected)) < upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err

    src = pa.gaussian_window(win_length, 0.5, sym, dtype=dtype).numpy()
    expected = get_window(('gaussian', 0.5), win_length, not sym)
    assert np.mean(np.abs(src - expected)) < upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err

    src = pa.exponential_window(win_length, None, 1.0, sym, dtype=dtype).numpy()
    expected = get_window(('exponential', None, 1.0), win_length, not sym)
    assert np.mean(np.abs(src - expected)) < upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err

    src = pa.taylor_window(win_length, 4, 30, True, sym, dtype=dtype).numpy()
    expected = get_window(('taylor', 4, 30, True), win_length, not sym)
    assert np.mean(np.abs(src - expected)) <= upper_err
    assert np.mean(np.abs(src - expected)) >= lower_err
