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
import paddleaudio
import pytest
import scipy

EPS = 1e-8
test_data = [
    (512, True),
    (512, False),
    (1024, True),
    (1024, False),
    (200, False),
    (200, True),
]


@pytest.mark.parametrize('win_length,sym', test_data)
def test_window(win_length, sym):

    assert np.allclose(paddleaudio.window.blackman(win_length, sym).numpy(),
                       scipy.signal.get_window('blackman', win_length, not sym),
                       atol=1e-6)
    assert np.allclose(paddleaudio.window.bohman(win_length, sym).numpy(),
                       scipy.signal.get_window('bohman', win_length, not sym),
                       atol=1e-6)
    assert np.allclose(paddleaudio.window.triang(win_length, sym).numpy(),
                       scipy.signal.get_window('triang', win_length, not sym),
                       atol=1e-6)
    assert np.allclose(paddleaudio.window.hamming(win_length, sym).numpy(),
                       scipy.signal.get_window('hamming', win_length, not sym),
                       atol=1e-6)
    assert np.allclose(paddleaudio.window.hann(win_length, sym).numpy(),
                       scipy.signal.get_window('hann', win_length, not sym),
                       atol=1e-6)

    assert np.allclose(paddleaudio.window.tukey(win_length, 0.5, sym).numpy(),
                       scipy.signal.get_window(('tukey', 0.5), win_length,
                                               not sym),
                       atol=1e-6)
    assert np.allclose(paddleaudio.window.gaussian(win_length, 0.5,
                                                   sym).numpy(),
                       scipy.signal.get_window(('gaussian', 0.5), win_length,
                                               not sym),
                       atol=1e-6)
    assert np.allclose(paddleaudio.window.exponential(win_length, None, 1.0,
                                                      sym).numpy(),
                       scipy.signal.get_window(('exponential', None, 1.0),
                                               win_length, not sym),
                       atol=1e-6)

    assert np.allclose(paddleaudio.window.taylor(win_length, 4, 30, True,
                                                 sym).numpy(),
                       scipy.signal.get_window(('taylor', 4, 30, True),
                                               win_length, not sym),
                       atol=1e-6)
