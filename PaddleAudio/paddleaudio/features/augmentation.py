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

from ..backends import depth_convert
from .utils import randint, weighted_sampling

__all__ = ['depth_augment', 'spect_augment', 'random_crop1d', 'random_crop2d']


# example y = depth_augment(y,['int8','int16'],[0.8,0.1])
def depth_augment(y, choices=['int8', 'int16'], probs=[0.5, 0.5]):
    assert len(probs) == len(choices), 'number of choices {} must be equal to size of probs {}'.format(
        len(choices), len(probs))
    k = weighted_sampling(probs)
    #k = randint(len(choices))
    src_depth = y.dtype
    y1 = depth_convert(y, choices[k])
    y2 = depth_convert(y1, src_depth)
    return y2


def adaptive_spect_augment(spect, tempo_axis=0, level=0.1):

    assert spect.ndim == 2., 'only supports 2d tensor or numpy array'
    if tempo_axis == 0:
        nt, nf = spect.shape
    else:
        nf, nt = spect.shape

    time_mask_width = int(nt * level * 0.5)
    freq_mask_width = int(nf * level * 0.5)

    num_time_mask = int(10 * level)
    num_freq_mask = int(10 * level)

    # num_zeros = num_time_mask*time_mask_width*nf + num_freq_mask*freq_mask_width*nt
    # factor = (nt*nf)/(nt*nf-num_zeros)

    if tempo_axis == 0:
        for i in range(num_time_mask):
            start = randint(nt - time_mask_width)
            spect[start:start + time_mask_width, :] = 0
        for i in range(num_freq_mask):
            start = randint(nf - freq_mask_width)
            spect[:, start:start + freq_mask_width] = 0
    else:
        for i in range(num_time_mask):
            start = randint(nt - time_mask_width)
            spect[:, start:start + time_mask_width] = 0
        for i in range(num_freq_mask):
            start = randint(nf - freq_mask_width)
            spect[start:start + freq_mask_width, :] = 0

    return spect


def spect_augment(
    spect,
    tempo_axis=0,
    max_time_mask=3,
    max_freq_mask=3,
    max_time_mask_width=30,
    max_freq_mask_width=20,
):

    assert spect.ndim == 2., 'only supports 2d tensor or numpy array'
    if tempo_axis == 0:
        nt, nf = spect.shape
    else:
        nf, nt = spect.shape

    num_time_mask = randint(max_time_mask)
    num_freq_mask = randint(max_freq_mask)

    time_mask_width = randint(max_time_mask_width)
    freq_mask_width = randint(max_freq_mask_width)

    #print(num_time_mask)
    #print(num_freq_mask)

    if tempo_axis == 0:
        for i in range(num_time_mask):
            start = randint(nt - time_mask_width)
            spect[start:start + time_mask_width, :] = 0
        for i in range(num_freq_mask):
            start = randint(nf - freq_mask_width)
            spect[:, start:start + freq_mask_width] = 0
    else:
        for i in range(num_time_mask):
            start = randint(nt - time_mask_width)
            spect[:, start:start + time_mask_width] = 0
        for i in range(num_freq_mask):
            start = randint(nf - freq_mask_width)
            spect[start:start + freq_mask_width, :] = 0

    return spect


def random_crop1d(y, crop_len):
    assert y.ndim == 1, 'only accept 1d tensor or numpy array'
    n = len(y)
    idx = randint(n - crop_len)
    return y[idx:idx + crop_len]


def random_crop2d(s, crop_len, tempo_axis=0):  # random crop according to temporal direction
    assert tempo_axis < s.ndim, 'axis out of range'
    n = s.shape[tempo_axis]
    idx = randint(high=n - crop_len)
    if type(s) == np.ndarray:
        sli = [slice(None) for i in range(s.ndim)]
        sli[tempo_axis] = slice(idx, idx + crop_len)
        out = s[tuple(sli)]
    else:
        out = paddle.index_select(s, paddle.Tensor(np.array([i for i in range(idx, idx + crop_len)])), axis=tempo_axis)
    return out
