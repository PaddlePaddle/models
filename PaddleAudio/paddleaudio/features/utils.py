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

__all__ = ['randint', 'rand', 'weighted_sampling']


def randint(high, use_paddle=True):
    if use_paddle:
        return int(paddle.randint(0, high=high))
    return int(np.random.randint(0, high=high))


def rand(use_paddle=True):
    if use_paddle:
        return float(paddle.rand((1, )))
    return float(np.random.rand(1))


def weighted_sampling(weights):
    n = len(weights)
    w = np.cumsum(weights)
    w = w / w[-1]
    flag = rand() < w
    return np.argwhere(flag)[0][0]
