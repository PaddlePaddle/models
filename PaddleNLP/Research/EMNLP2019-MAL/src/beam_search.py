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

import paddle.fluid as fluid
import paddle.fluid.layers as layers

INF = 1. * 1e9

class BeamSearch(object):
    """
        beam_search class
    """
    def __init__(self, beam_size, batch_size, alpha, vocab_size, hidden_size):
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.gather_top2k_append_index = layers.range(0, 2 * self.batch_size * beam_size, 1, 'int64') // \
                                                      (2 * self.beam_size) * (self.beam_size)

        self.gather_topk_append_index = layers.range(0, self.batch_size * beam_size, 1, 'int64') // \
                                                     self.beam_size * (2 * self.beam_size)

        self.gather_finish_topk_append_index = layers.range(0, self.batch_size * beam_size, 1, 'int64') // \
                                                            self.beam_size * (3 * self.beam_size)

        self.eos_id = layers.fill_constant([self.batch_size, 2 * self.beam_size], 'int64', value=1)
        self.get_alive_index = layers.range(0, self.batch_size, 1, 'int64') * self.beam_size

    
    def gather_cache(self, kv_caches, select_id):
        """
            gather cache
        """
        for index in xrange(len(kv_caches)):
            kv_cache = kv_caches[index]
            select_k = layers.gather(kv_cache['k'], [select_id])
            select_v = layers.gather(kv_cache['v'], [select_id])
            layers.assign(select_k, kv_caches[index]['k'])
            layers.assign(select_v, kv_caches[index]['v'])
    

    # topk_seq, topk_scores, topk_log_probs, topk_finished, cache
    def compute_topk_scores_and_seq(self, sequences, scores, scores_to_gather, flags, pick_finish=False, cache=None):
        """
            compute_topk_scores_and_seq
        """
        topk_scores, topk_indexes = layers.topk(scores, k=self.beam_size) #[batch_size, beam_size]
        if not pick_finish:
            flat_topk_indexes = layers.reshape(topk_indexes, [-1]) + self.gather_topk_append_index
            flat_sequences = layers.reshape(sequences, [2 * self.batch_size * self.beam_size, -1])
        else:
            flat_topk_indexes = layers.reshape(topk_indexes, [-1]) + self.gather_finish_topk_append_index
            flat_sequences = layers.reshape(sequences, [3 * self.batch_size * self.beam_size, -1])

        topk_seq = layers.gather(flat_sequences, [flat_topk_indexes])
        topk_seq = layers.reshape(topk_seq, [self.batch_size, self.beam_size, -1])

        flat_flags = layers.reshape(flags, [-1])
        topk_flags = layers.gather(flat_flags, [flat_topk_indexes])
        topk_flags = layers.reshape(topk_flags, [-1, self.beam_size])

        flat_scores = layers.reshape(scores_to_gather, [-1])
        topk_gathered_scores = layers.gather(flat_scores, [flat_topk_indexes]) 
        topk_gathered_scores = layers.reshape(topk_gathered_scores, [-1, self.beam_size])
        
        if cache:
            self.gather_cache(cache, flat_topk_indexes)

        return topk_seq, topk_gathered_scores, topk_flags, cache


    def grow_topk(self, i, logits, alive_seq, alive_log_probs, cache, enc_output, enc_bias):
        """
            grow_topk
        """
        logits = layers.reshape(logits, [self.batch_size, self.beam_size, -1])
        
        candidate_log_probs = layers.log(layers.softmax(logits, axis=2))
        log_probs = candidate_log_probs + layers.unsqueeze(alive_log_probs, axes=[2]) 
        
        base_1 = layers.cast(i, 'float32') + 6.0
        base_1 /= 6.0
        length_penalty = layers.pow(base_1, self.alpha)
        #length_penalty = layers.pow(((5.0 + layers.cast(i+1, 'float32')) / 6.0), self.alpha)
        
        curr_scores = log_probs / length_penalty
        flat_curr_scores = layers.reshape(curr_scores, [self.batch_size, self.beam_size * self.vocab_size])

        topk_scores, topk_ids = layers.topk(flat_curr_scores, k=self.beam_size * 2)
        
        topk_log_probs = topk_scores * length_penalty

        select_beam_index = topk_ids // self.vocab_size
        select_id = topk_ids % self.vocab_size

        #layers.Print(select_id, message="select_id", summarize=1024)
        #layers.Print(topk_scores, message="topk_scores", summarize=10000000)
        
        flat_select_beam_index = layers.reshape(select_beam_index, [-1]) + self.gather_top2k_append_index
        
        topk_seq = layers.gather(alive_seq, [flat_select_beam_index])
        topk_seq = layers.reshape(topk_seq, [self.batch_size, 2 * self.beam_size, -1])
        
        
        #concat with current ids
        topk_seq = layers.concat([topk_seq, layers.unsqueeze(select_id, axes=[2])], axis=2)
        topk_finished = layers.cast(layers.equal(select_id, self.eos_id), 'float32') 
        
        #gather cache
        self.gather_cache(cache, flat_select_beam_index)

        #topk_seq: [batch_size, 2*beam_size, i+1]
        #topk_log_probs, topk_scores, topk_finished: [batch_size, 2*beam_size]
        return topk_seq, topk_log_probs, topk_scores, topk_finished, cache


    def grow_alive(self, curr_seq, curr_scores, curr_log_probs, curr_finished, cache):
        """
            grow_alive
        """
        finish_float_flag = layers.cast(curr_finished, 'float32')
        finish_float_flag = finish_float_flag * -INF
        curr_scores += finish_float_flag

        return self.compute_topk_scores_and_seq(curr_seq, curr_scores, 
                                curr_log_probs, curr_finished, cache=cache)

    
    def grow_finished(self, i, finished_seq, finished_scores, finished_flags, curr_seq, 
                      curr_scores, curr_finished):
        """
            grow_finished
        """
        finished_seq = layers.concat([finished_seq, 
                                layers.fill_constant([self.batch_size, self.beam_size, 1], dtype='int64', value=0)], 
                                axis=2)

        curr_scores = curr_scores + (1.0 - layers.cast(curr_finished, 'int64')) * -INF

        curr_finished_seq = layers.concat([finished_seq, curr_seq], axis=1)
        curr_finished_scores = layers.concat([finished_scores, curr_scores], axis=1)
        curr_finished_flags = layers.concat([finished_flags, curr_finished], axis=1)
         
        return self.compute_topk_scores_and_seq(curr_finished_seq, curr_finished_scores, 
                                                curr_finished_scores, curr_finished_flags, 
                                                pick_finish=True)


    def inner_func(self, i, logits, alive_seq, alive_log_probs, finished_seq, finished_scores, 
                   finished_flags, cache, enc_output, enc_bias):
        """
            inner_func
        """
        topk_seq, topk_log_probs, topk_scores, topk_finished, cache = self.grow_topk(
                i, logits, alive_seq, alive_log_probs, cache, enc_output, enc_bias)

        alive_seq, alive_log_probs, _, cache = self.grow_alive(
                topk_seq, topk_scores, topk_log_probs, topk_finished, cache)
        #layers.Print(alive_seq, message="alive_seq", summarize=1024)

        finished_seq, finished_scores, finished_flags, _ = self.grow_finished(
                i, finished_seq, finished_scores, finished_flags, topk_seq, topk_scores, topk_finished)

        return alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags, cache


    def is_finished(self, step_idx, source_length, alive_log_probs, finished_scores, finished_in_finished):
        """
            is_finished
        """
        base_1 = layers.cast(source_length, 'float32') + 55.0
        base_1 /= 6.0
        max_length_penalty = layers.pow(base_1, self.alpha)

        flat_alive_log_probs = layers.reshape(alive_log_probs, [-1])
        lower_bound_alive_scores_1 = layers.gather(flat_alive_log_probs, [self.get_alive_index])
        
        lower_bound_alive_scores = lower_bound_alive_scores_1 / max_length_penalty
        
        lowest_score_of_finished_in_finish = layers.reduce_min(finished_scores * finished_in_finished, dim=1)

        finished_in_finished = layers.cast(finished_in_finished, 'bool')
        lowest_score_of_finished_in_finish += \
                        ((1.0 - layers.cast(layers.reduce_any(finished_in_finished, 1), 'float32')) * -INF)
        
        #print lowest_score_of_finished_in_finish
        bound_is_met = layers.reduce_all(layers.greater_than(lowest_score_of_finished_in_finish, 
                                                             lower_bound_alive_scores))

        decode_length = source_length + 50
        length_cond = layers.less_than(x=step_idx, y=decode_length)

        return layers.logical_and(x=layers.logical_not(bound_is_met), y=length_cond)
