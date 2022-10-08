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
import os

import numpy as np


class ReprodLogger(object):
    def __init__(self):
        self._data = dict()

    @property
    def data(self):
        return self._data

    def add(self, key, val):
        """
        添加key-val pair
        :param key:
        :param val:
        :return:
        """
        msg = '''val must be np.ndarray, you can convert it by follow code:
                1. Torch GPU: torch_tensor.cpu().detach().numpy()
                2. Torch CPU: torch_tensor.detach().numpy()
                3. Paddle: paddle_tensor.numpy()'''
        assert isinstance(val, np.ndarray), msg
        self._data[key] = val

    def remove(self, key):
        """
        移除key
        :param key:
        :return:
        """
        if key in self._data:
            self._data.pop(key)
        else:
            print('{} is not in {}'.format(key, self._data.keys()))

    def clear(self):
        """
        清空字典
        :return:
        """
        self._data.clear()

    def save(self, path):
        folder = os.path.dirname(path)
        if len(folder) >= 1:
            os.makedirs(folder, exist_ok=True)
        np.save(path, self._data)
