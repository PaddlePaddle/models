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
"""bert model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
import six
import logging
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.layers import shape

from model.transformer_encoder import encoder, pre_process_layer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logging.getLogger().setLevel(logging.INFO)                    
logger = logging.getLogger(__name__)

def dynamic_expand(dynamic_tensor, smaller_tensor):
    """
    :param dynamic_tensor:
    :param smaller_tensor:
    :return:
    """
    assert len(dynamic_tensor.shape) > len(smaller_tensor.shape)
    if type(smaller_tensor.shape) == list:
        for dim_idx, dim in smaller_tensor.shape:
            dynamic_tensor_dim_idx = len(dynamic_tensor) - len(smaller_tensor) + dim_idx
            assert dynamic_tensor.shape[dynamic_tensor_dim_idx] % dim == 0
    elif type(smaller_tensor.shape) == int:
        assert dynamic_tensor.shape[-1] % smaller_tensor.shape == 0
    memory_embs_zero = fluid.layers.scale(dynamic_tensor, scale=0.0)
    smaller_tensor = fluid.layers.elementwise_add(memory_embs_zero, smaller_tensor)
    return smaller_tensor


def print_tensor(tensor, message, print_runtime=False):
    logger.info("{}: {}".format(message, tensor.shape))
    if print_runtime:
        fluid.layers.Print(tensor, summarize=10, message=message)


class MemoryLayer(object):
    def __init__(self, bert_config, concept_size, mem_emb_size, mem_method='cat', prefix=None):
        self.initializer_range = bert_config['initializer_range']
        self.bert_size = bert_config['hidden_size']
        self.concept_size = concept_size
        self.mem_emb_size = mem_emb_size
        assert mem_method in ['add', 'cat', 'raw']
        self.mem_method = mem_method
        self.prefix = prefix

    def forward(self, bert_output, memory_embs, mem_length, ignore_no_memory_token=True):
        """
        :param bert_output: [batch_size, seq_size, bert_size]
        :param memory_embs: [batch_size, seq_size, concept_size, mem_emb_size]
        :param mem_length: [batch_size, sent_size, 1]
        :return: 
        """

        bert_size = self.bert_size
        concept_size = self.concept_size
        mem_emb_size = self.mem_emb_size

        print_tensor(bert_output, "bert_output")
        print_tensor(memory_embs, "memory_embs")
        print_tensor(mem_length, "mem_length")

 
        projected_bert = fluid.layers.fc(bert_output, size=mem_emb_size, num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name='{}_memory_layer_projection.w_0'.format(self.prefix) if self.prefix else 'memory_layer_projection.w_0',
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=self.initializer_range)),
            bias_attr=False)  # [batch_size *seq_size, mem_emb_size]
        logger.info("projected_bert: {}".format(projected_bert.shape))

        expanded_bert = fluid.layers.unsqueeze(projected_bert, axes=[2])   # [batch_size, seq_size, 1, mem_emb_size]

  
        extended_memory, memory_score = self.add_sentinel(expanded_bert, memory_embs, mem_emb_size)
        # extended_memory: [batch_size, seq_size, 1+concept_size, mem_emb_size]
        # memory_score: [batch_size, seq_size, 1+concept_size]


        concept_ordinal = self.get_concept_oridinal(concept_size, memory_score)  # [bs,sq,1+cs]

        memory_reverse_mask = fluid.layers.less_than(
            fluid.layers.expand(mem_length, expand_times=[1, 1, 1 + concept_size])
            , concept_ordinal)
        # [batch_size, seq_size, 1+concept_size]
        memory_reverse_mask = fluid.layers.cast(memory_reverse_mask, dtype="float32")
        print_tensor(memory_reverse_mask, "memory_reverse_mask")

        memory_reverse_masked_infinity = fluid.layers.scale(memory_reverse_mask, scale=-1e6)
        # [batch_size, seq_size, 1+concept_size]
        print_tensor(memory_reverse_masked_infinity, "memory_reverse_masked_infinity")

        memory_score = fluid.layers.elementwise_add(memory_score, memory_reverse_masked_infinity)
        # [batch_size, seq_size, 1+concept_size]
        logger.info("memory_score:{}".format(memory_score.shape))

        memory_att = fluid.layers.softmax(memory_score)  # [batch_size, seq_size, 1+concept_size]
        memory_att = fluid.layers.unsqueeze(memory_att, axes=[2])  # [batch_size, seq_size, 1, 1+concept_size]
        logger.info("memory_att: {}".format(memory_att.shape))
        logger.info("extended_memory: {}".format(extended_memory.shape))
        summ = fluid.layers.matmul(memory_att,extended_memory)  # [batch_size, seq_size,1, mem_emb_size]
        summ = fluid.layers.squeeze(summ, axes=[2])  # [batch_size, seq_size,mem_emb_size]

        if ignore_no_memory_token:
            condition = fluid.layers.less_than(
                dynamic_expand(mem_length, fluid.layers.zeros([1],"float32")),
                mem_length)  # [bs, sq]
            # summ_true = fluid.layers.elementwise_mul(
            #     summ,
            #     fluid.layers.cast(condition, "float32"))   # [bs, sq, ms]
            # summ_false = fluid.layers.elementwise_mul(
            #     summ,
            #     fluid.layers.scale(fluid.layers.cast(condition, "float32"), -1))  # [bs, sq, ms]
            # summ = fluid.layers.elementwise_add(summ_true, summ_false)  # [bs, sq, ms]
            summ = fluid.layers.elementwise_mul(
                summ,
                fluid.layers.cast(condition, "float32"))   # [bs, sq, ms]

            print_tensor(summ, "summ")

        if self.mem_method == "add":
            summ_transform = fluid.layers.fc(summ, size=bert_size, num_flatten_dims=2)  # [batch_size, seq_size, bert_size]
            output = fluid.layers.sums(input=[summ_transform, bert_output])  # [batch_size, seq_size, bert_size]
        elif self.mem_method == "cat":
            logger.info("bert_output: {}".format(bert_output.shape))
            logger.info("summ: {}".format(summ.shape))
            output = fluid.layers.concat(input=[bert_output, summ], axis=2)  # [batch_size, seq_size, bert_size + mem_emb_size]
        elif self.mem_method == "raw":
            logger.info("bert_output: {}".format(bert_output.shape))
            logger.info("summ: {}".format(summ.shape))
            output = summ  # [batch_size, seq_size, mem_emb_size]
        else:
            raise ValueError("mem_method not supported")
        logger.info("output: {}".format(output.shape))
        return output

    def get_concept_oridinal(self, concept_size, memory_score):
        """

        :param concept_size:
        :param memory_score: [batch_size, seq_size, 1+concept_size]
        :return:
        """
        concept_ordinal = fluid.layers.create_tensor(dtype="float32")
        fluid.layers.assign(np.arange(start=0, stop=(1 + concept_size), step=1, dtype=np.float32),
                            concept_ordinal)  # [1+cs]
        print_tensor(concept_ordinal, "concept_ordinal")
        print_tensor(memory_score, "memory_score")

        concept_ordinal = dynamic_expand(memory_score, concept_ordinal)  # [bs,sq,1+cs]

        logger.info("concept_ordinal: {}".format(concept_ordinal.shape))
        return concept_ordinal

    def add_sentinel(self, expanded_bert, memory_embs, mem_emb_size):
        """

        :param expanded_bert: [batch_size, seq_size, 1, mem_emb_size]
        :param memory_embs: [batch_size, seq_size, concept_size, mem_emb_size]
        :param mem_emb_size:
        :return:
        """
        sentinel = fluid.layers.create_parameter(
            name='{}_memory_layer_sentinel'.format(self.prefix) if self.prefix else 'memory_layer_sentinel',
            dtype="float32",
            shape=[mem_emb_size],
            default_initializer=fluid.initializer.ConstantInitializer(0))  # [mem_emb_size]
        print_tensor(sentinel, "sentinel")

        memory_embs_squeeze = fluid.layers.slice(memory_embs, axes=[2], starts=[0],
                                                 ends=[1])  # [bs,sq,1,ms]
        print_tensor(memory_embs_squeeze, "memory_embs_squeeze")

        sentinel = dynamic_expand(memory_embs_squeeze, sentinel)  # [bs,sq,1,ms]
        print_tensor(sentinel, "sentinel")
        print_tensor(memory_embs, "memory_embs")

        extended_memory = fluid.layers.concat([sentinel, memory_embs],
                                              axis=2)  # [batch_size, seq_size, 1+concept_size, mem_emb_size]
        extended_memory = fluid.layers.transpose(extended_memory, perm=[0, 1, 3, 2])
        # [batch_size, seq_size, mem_emb_size, 1+concept_size]
        logger.info("extended_memory: {}".format(extended_memory.shape))
        memory_score = fluid.layers.matmul(expanded_bert,
                                           extended_memory)  # [batch_size, seq_size, 1, 1+concept_size]
        memory_score = fluid.layers.squeeze(memory_score, axes=[2])
        # [batch_size, seq_size, 1+concept_size]
        extended_memory = fluid.layers.transpose(extended_memory, perm=[0, 1, 3, 2])
        # [batch_size, seq_size, 1+concept_size, mem_emb_size]
        return extended_memory, memory_score


class TriLinearTwoTimeSelfAttentionLayer(object):
    def __init__(self, hidden_size, dropout_rate=0.0,
    cat_mul=False, cat_sub=False, cat_twotime=False, cat_twotime_mul=False, cat_twotime_sub=False):
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.cat_mul = cat_mul
        self.cat_sub = cat_sub
        self.cat_twotime = cat_twotime
        self.cat_twotime_mul = cat_twotime_mul
        self.cat_twotime_sub = cat_twotime_sub

    def forward(self, hidden_emb, sequence_mask):
        """
        :param hidden_emb: [batch_size, seq_size, hidden_size]
        :param sequence_mask: [batch_size, seq_size, 1]
        :return:
        """
        assert len(hidden_emb.shape) ==3 and len(sequence_mask.shape) == 3 \
               and sequence_mask.shape[-1] == 1
        assert hidden_emb.shape[:2] == sequence_mask.shape[:2]  

        hidden_size = self.hidden_size

        bias = fluid.layers.create_parameter(name='self_matching_layer_bias', shape=[1], dtype="float32",
                        default_initializer=fluid.initializer.ConstantInitializer(0))

        weight_1 = fluid.layers.create_parameter(name='self_matching_layer_weight1', shape=[hidden_size], dtype="float32",
                        default_initializer=fluid.initializer.XavierInitializer(uniform=True, fan_in=1, fan_out=hidden_size))  # [HS]
        bs_1_hs = fluid.layers.slice(hidden_emb, axes=[1], starts=[0], ends=[1]) # [bs, 1, hs]
        print_tensor(bs_1_hs, "bs_1_hs")
        bs_hs_1 = fluid.layers.transpose(bs_1_hs, perm=[0, 2, 1])  # [bs, hs, 1]
        print_tensor(bs_hs_1, "bs_hs_1")
        print_tensor(weight_1, "weight_1")
        weight_1 = dynamic_expand(bs_1_hs, weight_1)  # [BS, 1, HS] (a)jk
        weight_1 = fluid.layers.transpose(weight_1, perm=[0, 2, 1])
        print_tensor(hidden_emb, "hidden_emb")
        print_tensor(weight_1, "weight_1")
        r1 = fluid.layers.matmul(hidden_emb, weight_1)  # [BS, SQ, 1]  aik
        print_tensor(r1, "r1")

        weight_2 = fluid.layers.create_parameter(name='self_matching_layer_weight2', shape=[hidden_size], dtype="float32",
                         default_initializer=fluid.initializer.XavierInitializer(uniform=True, fan_in=1, fan_out=hidden_size))  # [HS]
        weight_2 = dynamic_expand(bs_1_hs, weight_2)  # # [BS, 1, HS] (a)jk
        hidden_emb_transpose = fluid.layers.transpose(hidden_emb, perm=[0, 2, 1])  # [BS, HS, SQ] aji
        r2 = fluid.layers.matmul(weight_2, hidden_emb_transpose)  # [BS, 1, SQ]  aki
        print_tensor(r2, "r2")

        weight_mul = fluid.layers.create_parameter(name='self_matching_layer_weightmul', shape=[hidden_size], dtype="float32",
                        default_initializer=fluid.initializer.XavierInitializer(uniform=True))  # [HS]

 
        weight_mul = dynamic_expand(hidden_emb, weight_mul)
        rmul_1 = fluid.layers.elementwise_mul(hidden_emb, weight_mul)  # for "hidden * self.weight_mul". [bs, sq(i), hs(j)]
        print_tensor(rmul_1, "rmul_1")
        rmul_2 = fluid.layers.matmul(rmul_1, hidden_emb_transpose)  # [bs, sq(i), hs(j)] mul [bs, hs(j), sq(k)] = [bs, sq(i), sq(k)]
        print_tensor(rmul_2, "rmul_2")

        r1 = fluid.layers.squeeze(r1, axes=[2])  # [BS, SQ]  aik
        r1 = dynamic_expand(
            fluid.layers.transpose(rmul_2, [1, 0, 2]),  # [sq, bs, sq]
            r1)  # [ SQ(from 1), bs, SQ]
        r1 = fluid.layers.transpose(r1, [1, 2, 0])  # [bs, sq, sq(from 1)]

        r2 = fluid.layers.squeeze(r2, axes=[1])  # [BS, SQ]  aik
        r2 = dynamic_expand(
            fluid.layers.transpose(rmul_2, [1, 0, 2]),  # [sq, bs, sq]
            r2)  # [ SQ(from 1), bs, SQ]
        r2 = fluid.layers.transpose(r2, [1, 0, 2])  # [bs,sq(from 1),sq]

        bias = dynamic_expand(rmul_2, bias)  # [BS, SQ, SQ]
        sim_score = fluid.layers.sums(input=[r1, r2, rmul_2, bias])
        # [bs,sq,1]+[bs,1,sq]+[bs,sq,sq]+[bs,sq,sq]=[BS,SQ,SQ]
        print_tensor(sim_score, "sim_score")

        sequence_mask = fluid.layers.cast(sequence_mask, dtype="float32")  # [BS,SQ,1]
        softmax_mask = fluid.layers.elementwise_sub(
            sequence_mask,
            fluid.layers.fill_constant([1], "float32", 1))  # [BS,SQ,1]
        softmax_mask = fluid.layers.scale(softmax_mask, -1)
        very_negative_number = fluid.layers.fill_constant([1], value=-1e6, dtype="float32")
        logger.info("softmax_mask: {}".format(softmax_mask.shape))
        logger.info("very_negative_number: {}".format(very_negative_number.shape))

        softmax_mask = fluid.layers.elementwise_mul(softmax_mask, very_negative_number)  # [BS,SQ,1]

        softmax_mask = fluid.layers.squeeze(softmax_mask, axes=[2])  # [BS,SQ]
        softmax_mask = dynamic_expand(fluid.layers.transpose(sim_score, perm=[2, 0, 1]), softmax_mask)  # [sq(1),bs,sq]
        softmax_mask = fluid.layers.transpose(softmax_mask, perm=[1, 0, 2])   # [BS,sq(1),SQ]
        print_tensor(softmax_mask, "softmax_mask")
        sim_score = fluid.layers.elementwise_add(sim_score, softmax_mask)  # [bs,sq,sq]+[bs,sq(1),sq]=[BS,SQ,SQ]
        print_tensor(sim_score, "sim_score")

        attn_prob = fluid.layers.softmax(sim_score)  # [BS,SQ,SQ]
        weighted_sum = fluid.layers.matmul(attn_prob, hidden_emb)  # [bs,sq,sq]*[bs,sq,hs]=[BS,SQ,HS]
        if any([self.cat_twotime, self.cat_twotime_mul, self.cat_twotime_sub]):
            twotime_att_prob = fluid.layers.matmul(attn_prob, attn_prob)  # [bs,sq,sq]*[bs,sq,sq]=[BS,SQ,SQ]
            twotime_weited_sum = fluid.layers.matmul(twotime_att_prob, hidden_emb)  # [BS,SQ,HS]

        out_tensors = [hidden_emb, weighted_sum]
        if self.cat_mul:
            out_tensors.append(fluid.layers.elementwise_mul(hidden_emb, weighted_sum))
        if self.cat_sub:
            out_tensors.append(fluid.layers.elementwise_sub(hidden_emb, weighted_sum))
        if self.cat_twotime:
            out_tensors.append(twotime_weited_sum)
        if self.cat_twotime_mul:
            out_tensors.append(fluid.layers.elementwise_mul(hidden_emb, twotime_weited_sum))
        if self.cat_twotime_sub:
            out_tensors.append(fluid.layers.elementwise_sub(hidden_emb, twotime_weited_sum))
        output = fluid.layers.concat(out_tensors, axis=2)  # [BS,SQ, HS+HS+....]
        print_tensor(output, "output")
        return output



