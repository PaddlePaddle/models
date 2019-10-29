#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
Sampler class.
"""

import numpy as np


class Sampler(object):
    
    def __init__(self):
        return

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset
        return

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(range(len(self)))


class RandomSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset
        return

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(np.random.permutation(len(self)))
