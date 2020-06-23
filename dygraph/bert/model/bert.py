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
"dygraph transformer layers"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, to_variable, Layer, guard

from model.transformer_encoder import EncoderLayer, PrePostProcessLayer


class BertConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing bert model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class BertModelLayer(Layer):
    """
    bert
    """

    def __init__(self, config, return_pooled_out=True, use_fp16=False):
        super(BertModelLayer, self).__init__()

        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._sent_types = config['type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self.return_pooled_out = return_pooled_out

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._src_emb = Embedding(
            size=[self._voc_size, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._pos_emb = Embedding(
            size=[self._max_position_seq_len, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._sent_emb = Embedding(
            size=[self._sent_types, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self.pooled_fc = Linear(
            input_dim=self._emb_size,
            output_dim=self._emb_size,
            param_attr=fluid.ParamAttr(
                name="pooled_fc.w_0", initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0",
            act="tanh")

        self.pre_process_layer = PrePostProcessLayer(
            "nd", self._emb_size, self._prepostprocess_dropout, "")

        self._encoder = EncoderLayer(
            hidden_act=self._hidden_act,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer)

    def forward(self, src_ids, position_ids, sentence_ids, input_mask):
        """
        forward
        """
        src_emb = self._src_emb(src_ids)
        pos_emb = self._pos_emb(position_ids)
        sent_emb = self._sent_emb(sentence_ids)

        emb_out = src_emb + pos_emb
        emb_out = emb_out + sent_emb

        emb_out = self.pre_process_layer(emb_out)

        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)
        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask, scale=10000.0, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        enc_output = self._encoder(emb_out, n_head_self_attn_mask)

        if not self.return_pooled_out:
            return enc_output
        next_sent_feat = fluid.layers.slice(
            input=enc_output, axes=[1], starts=[0], ends=[1])
        next_sent_feat = self.pooled_fc(next_sent_feat)
        next_sent_feat = fluid.layers.reshape(
            next_sent_feat, shape=[-1, self._emb_size])

        return enc_output, next_sent_feat


class PretrainModelLayer(Layer):
    """
    pretrain model
    """

    def __init__(self,
                 config,
                 return_pooled_out=True,
                 weight_sharing=True,
                 use_fp16=False):
        super(PretrainModelLayer, self).__init__()
        self.config = config
        self._voc_size = config['vocab_size']
        self._emb_size = config['hidden_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']

        self._word_emb_name = "word_embedding"
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])
        self._weight_sharing = weight_sharing
        self.use_fp16 = use_fp16
        self._dtype = "float16" if use_fp16 else "float32"

        self.bert_layer = BertModelLayer(
            config=self.config, return_pooled_out=True, use_fp16=self.use_fp16)

        self.pre_process_layer = PrePostProcessLayer(
            "n", self._emb_size, self._prepostprocess_dropout, "pre_encoder")

        self.pooled_fc = Linear(
            input_dim=self._emb_size,
            output_dim=self._emb_size,
            param_attr=fluid.ParamAttr(
                name="mask_lm_trans_fc.w_0",
                initializer=self._param_initializer),
            bias_attr="mask_lm_trans_fc.b_0",
            act="tanh")

        self.mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        if not self._weight_sharing:
            self.out_fc = Linear(
                input_dim=self._emb_size,
                output_dim=self._voc_size,
                param_attr=fluid.ParamAttr(
                    name="mask_lm_out_fc.w_0",
                    initializer=self._param_initializer),
                bias_attr=self.mask_lm_out_bias_attr)
        else:
            self.fc_create_params = self.create_parameter(
                shape=[self._voc_size],
                dtype=self._dtype,
                attr=self.mask_lm_out_bias_attr,
                is_bias=True)

        self.next_sent_fc = Linear(
            input_dim=self._emb_size,
            output_dim=2,
            param_attr=fluid.ParamAttr(
                name="next_sent_fc.w_0", initializer=self._param_initializer),
            bias_attr="next_sent_fc.b_0")

    def forward(self, src_ids, position_ids, sentence_ids, input_mask,
                mask_label, mask_pos, labels):
        """
        forward
        """
        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        enc_output, next_sent_feat = self.bert_layer(src_ids, position_ids,
                                                     sentence_ids, input_mask)

        reshaped_emb_out = fluid.layers.reshape(
            x=enc_output, shape=[-1, self._emb_size])

        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)
        mask_trans_feat = self.pooled_fc(mask_feat)
        mask_trans_feat = self.pre_process_layer(mask_trans_feat)

        if self._weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=self.bert_layer._src_emb.weight,
                transpose_y=True)
            fc_out += self.fc_create_params
        else:
            fc_out = self.out_fc(mask_trans_feat)

        mask_lm_loss, mask_lm_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label, return_softmax=True)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        next_sent_fc_out = self.next_sent_fc(next_sent_feat)

        next_sent_loss, next_sent_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=next_sent_fc_out, label=labels, return_softmax=True)

        lm_acc = fluid.layers.accuracy(input=mask_lm_softmax, label=mask_label)

        next_sent_acc = fluid.layers.accuracy(
            input=next_sent_softmax, label=labels)
        mean_next_sent_loss = fluid.layers.mean(next_sent_loss)

        loss = mean_next_sent_loss + mean_mask_lm_loss
        return lm_acc, next_sent_acc, mean_mask_lm_loss, loss
