# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import time
import os
import ast
import glob
import yaml
import copy
import numpy as np


class Times(object):
    def __init__(self):
        self.time = 0.
        # start time
        self.st = 0.
        # end time
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, repeats=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / repeats
        else:
            self.time = (self.et - self.st) / repeats

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class PipeTimer(Times):
    def __init__(self, cfg):
        super(PipeTimer, self).__init__()
        self.total_time = Times()
        self.module_time = dict()
        for op in cfg:
            op_name = op.values()['name']
            self.module_time.update({op_name: Times()})

        self.img_num = 0

    def get_total_time(self):
        total_time = self.total_time.value()
        average_latency = total_time / max(1, self.img_num)
        qps = 0
        if total_time > 0:
            qps = 1 / average_latency
        return total_time, average_latency, qps

    def info(self):
        total_time, average_latency, qps = self.get_total_time()
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}".format(total_time * 1000,
                                                       self.img_num))

        for k, v in self.module_time.items():
            v_time = round(v.value(), 4)
            if v_time > 0:
                print("{} time(ms): {}; per frame average time(ms): {}".format(
                    k, v_time * 1000, v_time * 1000 / self.img_num))
        print("average latency time(ms): {:.2f}, QPS: {:2f}".format(
            average_latency * 1000, qps))
        return qps

    def report(self, average=False):
        dic = {}
        for m, time in self.module_time:
            dic[m] = round(time.value() / max(1, self.img_num),
                           4) if average else time.value()
        dic['total'] = round(self.total_time.value() / max(1, self.img_num),
                             4) if average else self.total_time.value()
        dic['img_num'] = self.img_num
        return dic
