# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# function:
#   transform samples in 'source' using 'mapper'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ..dataset import Dataset

class Transformer(Dataset):
    """ simple transformer without any workers to accelerate the processing
elerate                                                           
    """
    def __init__(self, source, mapper, worker_args=None):
        self._source = source
        self._mapper = mapper

    def next(self):
        sample = self._source.next()
        return self._mapper(sample)

    def reset(self):
        self._source.reset()

    def drained(self):
        return self._source.drained()

    def epoch_id(self):
        return self._source.epoch_id()

