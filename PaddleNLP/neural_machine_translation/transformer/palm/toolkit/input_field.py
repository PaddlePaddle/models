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

from __future__ import print_function
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

            self.append_placeholder(
                shape, dtype, lod_level=lod_level, name=name)

    def append_placeholder(self, shape, dtype, lod_level=0, name=""):
        self.shapes.append(shape)
        self.dtypes.append(dtype)
        self.lod_levels.append(lod_level)
        self.names.append(name)

    def build(self, capacity, reader_name, use_double_buffer=False):
        pyreader = fluid.layers.py_reader(
            capacity=capacity,
            shapes=self.shapes,
            dtypes=self.dtypes,
            lod_levels=self.lod_levels,
            name=reader_name,
            use_double_buffer=use_double_buffer)

        return [pyreader, fluid.layers.read_file(pyreader)]

    def __add__(self, new_holder):
        assert isinstance(new_holder, tuple) or isinstance(new_holder, list)
        assert len(new_holder) >= 2

        shape = new_holder[0]
        dtype = new_holder[1]
        lod_level = new_holder[2] if len(new_holder) >= 3 else 0
        name = new_holder[3] if len(new_holder) >= 4 else ""

        self.append_placeholder(shape, dtype, lod_level=lod_level, name=name)


class InputField(object):
    """
    A high-level API for handling inputs in PaddlePaddle.
    """

    def __init__(self, input_slots=[]):

        self.shapes = []
        self.dtypes = []
        self.names = []
        self.lod_levels = []

        self.input_slots = {}
        self.feed_list_str = []
        self.feed_list = []

        self.reader = None

        if input_slots:
            for input_slot in input_slots:
                self += input_slot

    def __add__(self, input_slot):

        if isinstance(input_slot, list) or isinstance(input_slot, tuple):
            name = input_slot[0]
            shape = input_slot[1]
            dtype = input_slot[2]
            lod_level = input_slot[3] if len(input_slot) == 4 else 0

        if isinstance(input_slot, dict):
            name = input_slot["name"]
            shape = input_slot["shape"]
            dtype = input_slot["dtype"]
            lod_level = input_slot[
                "lod_level"] if "lod_level" in input_slot else 0

        self.shapes.append(shape)
        self.dtypes.append(dtype)
        self.names.append(name)
        self.lod_levels.append(lod_level)

        self.feed_list_str.append(name)

        return self

    def __getattr__(self, name):

        if name not in self.input_slots:
            raise Warning("the attr %s has not been defined yet." % name)
            return None

        return self.input_slots[name]

    def build(self, build_pyreader=False, capacity=100, iterable=False):

        for _name, _shape, _dtype, _lod_level in zip(
                self.names, self.shapes, self.dtypes, self.lod_levels):
            self.input_slots[_name] = fluid.layers.data(
                name=_name, shape=_shape, dtype=_dtype, lod_level=_lod_level)

        for name in self.feed_list_str:
            self.feed_list.append(self.input_slots[name])

        if build_pyreader:
            self.reader = fluid.io.PyReader(
                feed_list=self.feed_list, capacity=capacity, iterable=iterable)

    def start(self, generator=None):

        if generator is not None:
            self.reader.decorate_batch_generator(generator)

        self.reader.start()


if __name__ == "__main__":

    mnist_input_slots = [{
        "name": "image",
        "shape": (-1, 32, 32, 1),
        "dtype": "int32"
    }, {
        "name": "label",
        "shape": [-1, 1],
        "dtype": "int64"
    }]

    input_field = InputField(mnist_input_slots)

    input_field += {
        "name": "large_image",
        "shape": (-1, 64, 64, 1),
        "dtype": "int32"
    }
    input_field += {
        "name": "large_color_image",
        "shape": (-1, 64, 64, 3),
        "dtype": "int32"
    }

    input_field.build()

    print(input_field.feed_list)

    print(input_field.image)

    print(input_field.large_color_image)
