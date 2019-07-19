#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
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
######################################################################
"""
File: interact.py
"""

import paddle.fluid as fluid
import paddle.fluid.framework as framework
from source.inputters.data_provider import load_dict
from source.inputters.data_provider import MatchProcessor
from source.inputters.data_provider import preprocessing_for_one_line

import numpy as np

load_dict("./dict/gene.dict")

def load_model(): 
    """
    load model function
    """
    main_program = fluid.default_main_program()
    #place = fluid.CPUPlace()
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(framework.default_startup_program())

    path = "./models/inference_model"
    [inference_program, feed_dict, fetch_targets] = \
        fluid.io.load_inference_model(dirname=path, executor=exe)
    model_handle = [exe, inference_program, feed_dict, fetch_targets, place]
    return model_handle


def predict(model_handle, text, task_name):
    """
    predict score function
    """
    exe = model_handle[0]
    inference_program = model_handle[1]
    feed_dict = model_handle[2]
    fetch_targets = model_handle[3]
    place = model_handle[4]

    data = preprocessing_for_one_line(text, MatchProcessor.get_labels(), \
                                      task_name, max_seq_len=256)
    context_ids = [elem[0] for elem in data]
    context_pos_ids = [elem[1] for elem in data]
    context_segment_ids = [elem[2] for elem in data]
    context_attn_mask = [elem[3] for elem in data]
    labels_ids = [[1]]
    if 'kn' in task_name:
        kn_ids = [elem[4] for elem in data]
        kn_ids = fluid.create_lod_tensor(kn_ids, [[len(kn_ids[0])]], place)
        context_next_sent_index = [elem[5] for elem in data]
        results = exe.run(inference_program,
                        feed={feed_dict[0]: np.array(context_ids),
                              feed_dict[1]: np.array(context_pos_ids),
                              feed_dict[2]: np.array(context_segment_ids),
                              feed_dict[3]: np.array(context_attn_mask),
                              feed_dict[4]: kn_ids, 
                              feed_dict[5]: np.array(labels_ids),
                              feed_dict[6]: np.array(context_next_sent_index)},
                        fetch_list=fetch_targets)
    else:
        context_next_sent_index = [elem[4] for elem in data]
        results = exe.run(inference_program,
                          feed={feed_dict[0]: np.array(context_ids),
                                feed_dict[1]: np.array(context_pos_ids),
                                feed_dict[2]: np.array(context_segment_ids),
                                feed_dict[3]: np.array(context_attn_mask),
                                feed_dict[4]: np.array(labels_ids),
                                feed_dict[5]: np.array(context_next_sent_index)},
                          fetch_list=fetch_targets)
    score = results[0][0][1]
    return score

