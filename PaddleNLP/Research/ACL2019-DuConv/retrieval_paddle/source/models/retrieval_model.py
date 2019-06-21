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
File: retrieval_model.py
"""

import six
import json
import numpy as np
import paddle.fluid as fluid
from source.encoders.transformer import encoder, pre_process_layer



class RetrievalModel(object):
    def __init__(self,
                 context_ids,
                 context_pos_ids,
                 context_segment_ids,
                 context_attn_mask,
                 kn_ids,
                 emb_size=1024,
                 n_layer=12,
                 n_head=1,
                 voc_size=10005,
                 max_position_seq_len=512,
                 sent_types=2,
                 hidden_act='relu',
                 prepostprocess_dropout=0.1,
                 attention_dropout=0.1,
                 weight_sharing=True):
        self._emb_size = emb_size
        self._n_layer = n_layer
        self._n_head = n_head
        self._voc_size = voc_size
        self._sent_types = sent_types
        self._max_position_seq_len = max_position_seq_len
        self._hidden_act = hidden_act
        self._weight_sharing = weight_sharing
        self._prepostprocess_dropout = prepostprocess_dropout
        self._attention_dropout = attention_dropout

        self._context_emb_name = "context_word_embedding"
        self._memory_emb_name = "memory_word_embedding"
        self._context_pos_emb_name = "context_pos_embedding"
        self._context_segment_emb_name = "context_segment_embedding"
        if kn_ids: 
            self._memory_emb_name = "memory_word_embedding"
        self._build_model(context_ids, context_pos_ids, \
                context_segment_ids, context_attn_mask, kn_ids)

    def _build_memory_network(self, kn_ids, rnn_hidden_size=128): 
        kn_emb_out = fluid.layers.embedding(
                input=kn_ids,
                size=[self._voc_size, self._emb_size],
                dtype='float32')
        para_attr = fluid.ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.02))
        bias_attr = fluid.ParamAttr(
                        initializer=fluid.initializer.Normal(0.0, 0.02))

        fc_fw = fluid.layers.fc(input=kn_emb_out,
                    size=rnn_hidden_size * 3,
                    param_attr=para_attr,
                    bias_attr=False)
        fc_bw = fluid.layers.fc(input=kn_emb_out,
                    size=rnn_hidden_size * 3,
                    param_attr=para_attr,
                    bias_attr=False)
        gru_forward = fluid.layers.dynamic_gru(
                    input=fc_fw,
                    size=rnn_hidden_size,
                    param_attr=para_attr,
                    bias_attr=bias_attr,
                    candidate_activation='relu')
        gru_backward = fluid.layers.dynamic_gru(
                    input=fc_bw,
                    size=rnn_hidden_size,
                    is_reverse=True,
                    param_attr=para_attr,
                    bias_attr=bias_attr,
                    candidate_activation='relu')

        memory_encoder_out = fluid.layers.concat(
                        input=[gru_forward, gru_backward], axis=1)

        memory_encoder_proj_out = fluid.layers.fc(input=memory_encoder_out,
                        size=256,
                        bias_attr=False)
        return memory_encoder_out, memory_encoder_proj_out

    def _build_model(self, 
                    context_ids, 
                    context_pos_ids, 
                    context_segment_ids, 
                    context_attn_mask, 
                    kn_ids): 
        
        context_emb_out = fluid.layers.embedding(
            input=context_ids,
            size=[self._voc_size, self._emb_size],
            param_attr=fluid.ParamAttr(name=self._context_emb_name),
            is_sparse=False)
        
        context_position_emb_out = fluid.layers.embedding(
            input=context_pos_ids,
            size=[self._max_position_seq_len, self._emb_size],
            param_attr=fluid.ParamAttr(name=self._context_pos_emb_name), )

        context_segment_emb_out = fluid.layers.embedding(
            input=context_segment_ids, 
            size=[self._sent_types, self._emb_size],
            param_attr=fluid.ParamAttr(name=self._context_segment_emb_name), )

        context_emb_out = context_emb_out + context_position_emb_out
        context_emb_out = context_emb_out + context_segment_emb_out

        context_emb_out = pre_process_layer(
            context_emb_out, 'nd', self._prepostprocess_dropout, name='context_pre_encoder')

        n_head_context_attn_mask = fluid.layers.stack(
            x=[context_attn_mask] * self._n_head, axis=1)

        n_head_context_attn_mask.stop_gradient = True

        self._context_enc_out = encoder(
            enc_input=context_emb_out,
            attn_bias=n_head_context_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="an",
            postprocess_cmd="dan",
            name='context_encoder')
       
        if kn_ids: 
            self.memory_encoder_out, self.memory_encoder_proj_out = \
                    self._build_memory_network(kn_ids)

    def get_context_output(self, context_next_sent_index, task_name): 
        if "kn" in task_name: 
            cls_feats = self.get_context_response_memory(context_next_sent_index)
        else: 
            cls_feats = self.get_pooled_output(context_next_sent_index)
        return cls_feats

    def get_context_response_memory(self, context_next_sent_index): 
        context_out = self.get_pooled_output(context_next_sent_index)
        kn_context = self.attention(context_out, \
                self.memory_encoder_out, self.memory_encoder_proj_out)
        cls_feats = fluid.layers.concat(input=[context_out, kn_context], axis=1)
        return cls_feats

    def attention(self, hidden_mem, encoder_vec, encoder_vec_proj): 
        concated = fluid.layers.sequence_expand(
                        x=hidden_mem, y=encoder_vec_proj)

        concated = encoder_vec_proj + concated
        concated = fluid.layers.tanh(x=concated)
        attention_weights = fluid.layers.fc(input=concated,
                                size=1,
                                act=None,
                                bias_attr=False)
        attention_weights = fluid.layers.sequence_softmax(
                                input=attention_weights)
        weigths_reshape = fluid.layers.reshape(x=attention_weights, shape=[-1])
        scaled = fluid.layers.elementwise_mul(
                    x=encoder_vec, y=weigths_reshape, axis=0)
        context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
        return context

    def get_sequence_output(self):
        return (self._context_enc_out, self._response_enc_out)

    def get_pooled_output(self, context_next_sent_index): 
        context_out = self.get_pooled(context_next_sent_index)
        return context_out

    def get_pooled(self, next_sent_index):
        """Get the first feature of each sequence for classification"""
        reshaped_emb_out = fluid.layers.reshape(
            x=self._context_enc_out, shape=[-1, self._emb_size], inplace=True)
        next_sent_index = fluid.layers.cast(x=next_sent_index, dtype='int32')
        next_sent_feat = fluid.layers.gather(
            input=reshaped_emb_out, index=next_sent_index)
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat
        

    def get_pooled_output_no_share(self, context_next_sent_index, response_next_sent_index):
        """get pooled embedding"""
        self._context_reshaped_emb_out = fluid.layers.reshape(
            x=self._context_enc_out, shape=[-1, self._emb_size], inplace=True)
        context_next_sent_index = fluid.layers.cast(x=context_next_sent_index, dtype='int32')
        context_out = fluid.layers.gather(
            input=self._context_reshaped_emb_out, index=context_next_sent_index)
        context_out = fluid.layers.fc(
            input=context_out,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_context_fc.w_0",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr="pooled_context_fc.b_0")

        self._response_reshaped_emb_out = fluid.layers.reshape(
            x=self._response_enc_out, shape=[-1, self._emb_size], inplace=True)
        response_next_sent_index = fluid.layers.cast(x=response_next_sent_index, dtype='int32')
        response_next_sent_feat = fluid.layers.gather(
            input=self._response_reshaped_emb_out, index=response_next_sent_index)
        response_next_sent_feat = fluid.layers.fc(
            input=response_next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(
                name="pooled_response_fc.w_0",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr="pooled_response_fc.b_0")
        
        return context_out, response_next_sent_feat


