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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import ast
import copy

import numpy as np
import paddle.fluid as fluid


class Placeholder(object):

    def __init__(self):
        self.shapes = []
        self.dtypes = []
        self.lod_levels = []
        self.names = []

    def __init__(self, input_shapes):

        self.shapes = []
        self.dtypes = []
        self.lod_levels = []
        self.names = []

        for new_holder in input_shapes:
            shape = new_holder[0]
            dtype = new_holder[1]
            lod_level = new_holder[2] if len(new_holder) >= 3 else 0
            name = new_holder[3] if len(new_holder) >= 4 else ""

            self.append_placeholder(shape, dtype, lod_level = lod_level, name = name)

    def append_placeholder(self, shape, dtype, lod_level = 0, name = ""):
        self.shapes.append(shape)
        self.dtypes.append(dtype)
        self.lod_levels.append(lod_level)
        self.names.append(name)

    def build(self, capacity, reader_name, use_double_buffer = False):
        pyreader = fluid.layers.py_reader(
            capacity = capacity,
            shapes = self.shapes,
            dtypes = self.dtypes,
            lod_levels = self.lod_levels,
            name = reader_name, 
            use_double_buffer = use_double_buffer)

        return [pyreader, fluid.layers.read_file(pyreader)]

    def __add__(self, new_holder):
        assert isinstance(new_holder, tuple) or isinstance(new_holder, list) 
        assert len(new_holder) >= 2

        shape = new_holder[0]
        dtype = new_holder[1]
        lod_level = new_holder[2] if len(new_holder) >= 3 else 0
        name = new_holder[3] if len(new_holder) >= 4 else ""

        self.append_placeholder(shape, dtype, lod_level = lod_level, name = name)


if __name__ == "__main__":
    print("hello world!")



