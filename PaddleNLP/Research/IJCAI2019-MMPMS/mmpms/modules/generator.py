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

import numpy as np

import paddle.fluid as fluid
import mmpms.layers as layers


def state_assign(new_state, old_state):
    if isinstance(new_state, dict):
        for k in new_state.keys():
            state_assign(new_state[k], old_state[k])
    elif isinstance(new_state, (tuple, list)):
        assert len(new_state) == len(old_state)
        for new_s, old_s in zip(new_state, old_state):
            state_assign(new_s, old_s)
    else:
        layers.assign(new_state, old_state)


def state_sequence_expand(state, y):
    if isinstance(state, dict):
        return {k: state_sequence_expand(v, y) for k, v in state.items()}
    elif isinstance(state, (tuple, list)):
        return type(state)(state_sequence_expand(s, y) for s in state)
    else:
        if state.dtype != y.dtype:
            return layers.sequence_expand(state, layers.cast(y, state.dtype))
        else:
            return layers.sequence_expand(state, y)


class BeamSearch(object):
    def __init__(self,
                 vocab_size,
                 beam_size,
                 start_id,
                 end_id,
                 unk_id,
                 min_length=1,
                 max_length=30,
                 length_average=False,
                 ignore_unk=False,
                 ignore_repeat=False):
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.start_id = start_id
        self.end_id = end_id
        self.unk_id = unk_id
        self.min_length = min_length
        self.max_length = max_length
        self.length_average = length_average
        self.ignore_unk = ignore_unk
        self.ignore_repeat = ignore_repeat

    def __call__(self, step_fn, state, init_ids):

        init_scores = layers.fill_constant_batch_size_like(
            input=init_ids, shape=[-1, 1], dtype="float32", value=0)
        init_scores = layers.lod_reset(init_scores, init_ids)

        unk_scores = np.zeros(self.vocab_size, dtype="float32")
        unk_scores[self.unk_id] = -1e9
        unk_scores = layers.assign(unk_scores)

        end_scores = np.zeros(self.vocab_size, dtype="float32")
        end_scores[self.end_id] = -1e9
        end_scores = layers.assign(end_scores)

        array_len = layers.fill_constant(
            shape=[1], dtype="int64", value=self.max_length)
        min_array_len = layers.fill_constant(
            shape=[1], dtype="int64", value=self.min_length)
        counter = layers.zeros(shape=[1], dtype="int64", force_cpu=True)

        # ids, scores as memory
        ids_array = layers.create_array("int64")
        scores_array = layers.create_array("float32")

        layers.array_write(init_ids, array=ids_array, i=counter)
        layers.array_write(init_scores, array=scores_array, i=counter)

        cond = layers.less_than(x=counter, y=array_len)
        while_op = layers.While(cond=cond)

        with while_op.block():
            pre_ids = layers.array_read(array=ids_array, i=counter)
            pre_score = layers.array_read(array=scores_array, i=counter)

            # use step_fn to update state and get score
            score, new_state = step_fn(pre_ids, state)
            score = layers.log(score)

            if self.ignore_unk:
                score = score + unk_scores

            if self.ignore_repeat:
                repeat_scores = layers.cast(
                    layers.one_hot(pre_ids, self.vocab_size), "float32") * -1e9
                score = score + repeat_scores

            min_cond = layers.less_than(x=counter, y=min_array_len)
            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(min_cond):
                    layers.assign(score + end_scores, score)

            score = layers.lod_reset(x=score, y=pre_score)

            topk_scores, topk_indices = layers.topk(score, k=self.beam_size)
            if self.length_average:
                pre_num = layers.cast(counter, "float32")
                cur_num = layers.increment(pre_num, value=1.0, in_place=False)
                accu_scores = layers.elementwise_add(
                    x=layers.elementwise_div(topk_scores, cur_num),
                    y=layers.elementwise_div(
                        layers.elementwise_mul(
                            layers.reshape(
                                pre_score, shape=[-1]), pre_num),
                        cur_num),
                    axis=0)
            else:
                accu_scores = layers.elementwise_add(
                    x=topk_scores,
                    y=layers.reshape(
                        pre_score, shape=[-1]),
                    axis=0)

            selected_ids, selected_scores, parent_idx = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_score,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=self.beam_size,
                end_id=self.end_id,
                return_parent_idx=True)

            layers.increment(x=counter, value=1, in_place=True)

            # update the memories
            layers.array_write(selected_ids, array=ids_array, i=counter)
            layers.array_write(selected_scores, array=scores_array, i=counter)
            state_assign(new_state, state)

            length_cond = layers.less_than(x=counter, y=array_len)
            not_finish_cond = layers.logical_not(
                layers.is_empty(x=selected_ids))
            layers.logical_and(x=length_cond, y=not_finish_cond, out=cond)

            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(not_finish_cond):
                    new_state = state_sequence_expand(new_state,
                                                      selected_scores)
                    state_assign(new_state, state)

        prediction_ids, prediction_scores = layers.beam_search_decode(
            ids=ids_array,
            scores=scores_array,
            beam_size=self.beam_size,
            end_id=self.end_id)
        return prediction_ids, prediction_scores
