#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
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
################################################################################

import paddle.fluid as fluid


class Model(object):
    def __init__(self, use_gpu=False):
        self.train_program = None
        self.eval_program = None
        self.infer_program = None
        self.startup_program = None
        self.train_fetch_dict = None
        self.eval_fetch_dict = None
        self.infer_fetch_dict = None

        self.build_program()

        assert self.startup_program is not None
        assert self.train_program is not None
        assert self.train_fetch_dict is not None
        assert self.eval_program is not None
        assert self.eval_fetch_dict is not None

        self.place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

        self.executor = fluid.Executor(self.place)
        self.executor.run(self.startup_program)

    def build_program(self):
        raise NotImplementedError

    def set_feed(self, inputs, mode):
        raise NotImplementedError

    def train(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs):
        raise NotImplementedError

    def infer(self, inputs):
        raise NotImplementedError

    def execute(self, program, feed, fetch_dict, return_numpy=True):
        fetch_keys = list(fetch_dict.keys())
        fetch_list = list(fetch_dict.values())
        fetch_vals = self.executor.run(program=program,
                                       feed=feed,
                                       fetch_list=fetch_list,
                                       return_numpy=return_numpy)
        return dict(zip(fetch_keys, fetch_vals))

    def save(self, model_dir):
        """ Save model parameters. """
        fluid.io.save_persistables(
            executor=self.executor,
            dirname=model_dir,
            main_program=self.train_program)

    def load(self, model_dir):
        """ Load model parameters. """
        fluid.io.load_persistables(
            executor=self.executor,
            dirname=model_dir,
            main_program=self.train_program)
