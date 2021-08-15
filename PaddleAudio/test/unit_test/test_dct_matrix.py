# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import scipy
import utils


def test_dct_compat_with_scipy1():

    paddle.set_device('cpu')
    expected = scipy.fft.dct(np.eye(64), norm='ortho')[:, :8]
    paddle_dct = paddleaudio.functional.dct_matrix(8, 64, dct_norm='ortho')
    err = np.mean(np.abs(paddle_dct.numpy() - expected))
    assert err < 5e-8


def test_dct_compat_with_scipy2():

    paddle.set_device('cpu')
    expected = scipy.fft.dct(np.eye(64), norm=None)[:, :8]
    paddle_dct = paddleaudio.functional.dct_matrix(8, 64, dct_norm=None)
    err = np.mean(np.abs(paddle_dct.numpy() - expected))
    assert err < 5e-7


def test_dct_compat_with_scipy3():

    paddle.set_device('gpu')
    expected = scipy.fft.dct(np.eye(64), norm='ortho')[:, :8]
    paddle_dct = paddleaudio.functional.dct_matrix(8, 64, dct_norm='ortho')
    err = np.mean(np.abs(paddle_dct.numpy() - expected))
    assert err < 5e-7


def test_dct_compat_with_scipy4():

    paddle.set_device('gpu')
    expected = scipy.fft.dct(np.eye(64), norm=None)[:, :8]
    paddle_dct = paddleaudio.functional.dct_matrix(8, 64, dct_norm=None)
    err = np.mean(np.abs(paddle_dct.numpy() - expected))
    assert err < 5e-7
