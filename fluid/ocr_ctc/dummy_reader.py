"""A dummy reader for test."""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
DATA_SHAPE = [1, 512, 512]
NUM_CLASSES = 20


def _read_creater(num_sample=1024, min_seq_len=1, max_seq_len=10):
    def reader():
        for i in range(num_sample):
            sequence_len = np.random.randint(min_seq_len, max_seq_len)
            x = np.random.uniform(0.1, 1, DATA_SHAPE).astype("float32")
            y = np.random.randint(0, NUM_CLASSES + 1,
                                  [sequence_len]).astype("int32")
            yield x, y

    return reader


def train(num_sample=16):
    """Get train dataset reader."""
    return _read_creater(num_sample=num_sample)


def test(num_sample=16):
    """Get test dataset reader."""
    return _read_creater(num_sample=num_sample)


def data_shape():
    """Get image shape in CHW order."""
    return DATA_SHAPE


def num_classes():
    """Get number of total classes."""
    return NUM_CLASSES
