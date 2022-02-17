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
"""Common utils functions for testing """
import os

import librosa
import numpy as np
from paddle.utils import download

AUDIO_URL1 = 'https://bj.bcebos.com/paddleaudio/test/data/test_audio.wav'


def relative_err(a, b, real=True):
    """compute relative error of two matrices or vectors"""
    if real:
        return np.sum((a - b)**2) / (1e-8 + np.sum(a**2) + np.sum(b**2))
    else:
        err = np.sum((a.real-b.real)**2) / \
            (1e-8+np.sum(a.real**2)+np.sum(b.real**2))
        err += np.sum((a.imag-b.imag)**2) / \
            (1e-8+np.sum(a.imag**2)+np.sum(b.imag**2))

        return err


def load_example_audio1(sr=16000):
    file = download.get_weights_path_from_url(AUDIO_URL1)
    x, r = librosa.load(file, sr=sr)
    return x, r


def get_example_audio1() -> os.PathLike:
    file = download.get_weights_path_from_url(AUDIO_URL1)
    return file
