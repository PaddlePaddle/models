# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math

from .. import PretrainedModel, register_base_model
from paddle.nn.layer.transformer import _convert_param_attr_to_list

__all__ = [
    'GPT2Model',
    "GPT2PretrainedModel",
    'GPT2ForPretraining',
    'GPT2PretrainingCriterion',
    # 'GPT2PretrainingHeads',
    'GPT2ForQuestionAnswering',
]


class TransformerDecoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=True,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = nn.MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        print("Fuck the origin epsilon")
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        if not isinstance(cache, self.self_attn.Cache):
            pk, pv = paddle.unstack(cache, axis=1)
            cache = self.self_attn.gen_cache(pk, pv)
        print("tgt", tgt)
        print("tgt sum", paddle.sum(tgt))
        print("in norm sum", paddle.sum(self.norm1.weight))
        print("in norm bias sum", paddle.sum(self.norm1.bias))
        import os
        import numpy as np
        if not os.path.exists("./new_tgt.npy"):
            np.save("./new_tgt.npy", tgt.numpy())
            np.save("./new_norm_w.npy", self.norm1.weight.numpy())
            np.save("./new_norm_b.npy", self.norm1.bias.numpy())
            print("in norm shape", self.norm1._normalized_shape)
            print("in norm epsilon", self.norm1._epsilon)

        residual = tgt
        if self.normalize_before:
            print("fuck tgt", tgt)
            tgt = self.norm1(tgt)
            if not os.path.exists("./new_tgt.npy"):
                np.save("./new_normed_tgt.npy", tgt.numpy())
            print("fuck layer norm", tgt)

        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache)
        print("attn", tgt)
        # Dropout ?
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        print("before mlp", tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        print("after mlp", tgt)
        tgt = residual + self.dropout2(tgt)

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        return tgt if cache is None else (tgt, incremental_cache)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(
            memory, type=self.self_attn.Cache)
        return incremental_cache


class GPT2Embeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16):
        super(GPT2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones

        print("position_ids", position_ids)
        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embedings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class GPT2PretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained GPT2 models. It provides GPT2 related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "gpt2-base": {
            "vocab_size": 30000,
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 10240,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
    }
    #resource_files_names = {"model_state": "model_state.pdparams"}
    #pretrained_resource_files_map = {
    #    "model_state": {
    #        "gpt2-base-uncased":
    #        "https://paddlenlp.bj.bcebos.com/models/transformers/gpt2-base-uncased.pdparams",
    #    }
    #}
    base_model_prefix = "gpt2"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.gpt2.config["initializer_range"],
                        shape=layer.weight.shape))
        # elif isinstance(layer, nn.LayerNorm):
        #     layer._epsilon = 1e-12


@register_base_model
class GPT2Model(GPT2PretrainedModel):
    """
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0):
        super(GPT2Model, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = GPT2Embeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_hidden_layers, norm=nn.LayerNorm(hidden_size))
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                kv_cache=None):
        if attention_mask is None:
            # attention_mask = paddle.unsqueeze(
            #     (input_ids == self.pad_token_id
            #      ).astype(self.embeddings.word_embeddings.weight.dtype) * -1e9,
            #     axis=[1, 2])
            length = input_ids.shape[1]
            attention_mask = paddle.tensor.triu(
                (paddle.ones(
                    (length, length),
                    dtype=self.embeddings.word_embeddings.weight.dtype) * -1e9),
                1)
        if position_ids is None and kv_cache is not None:
            past_length = kv_cache[0][0].shape[-2]
            position_ids = paddle.arange(
                past_length, input_ids.shape[-1] + past_length, dtype='int64')
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        # attention_mask
        print("embedding_output", embedding_output)
        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            cache=kv_cache)
        return encoder_outputs


class GPT2ForQuestionAnswering(GPT2PretrainedModel):
    def __init__(self, gpt2, dropout=None):
        super(GPT2ForQuestionAnswering, self).__init__()
        self.gpt2 = gpt2  # allow gpt2 to be config
        self.classifier = nn.Linear(self.gpt2.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None):
        sequence_output, _ = self.gpt2(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


# class GPT2LMPredictionHead(nn.Layer):
#     def __init__(self,
#                  hidden_size,
#                  vocab_size,
#                  activation,
#                  embedding_weights=None):
#         super(GPT2LMPredictionHead, self).__init__()
#         self.transform = nn.Linear(hidden_size, hidden_size)
#         self.activation = getattr(nn.functional, activation)
#         self.layer_norm = nn.LayerNorm(hidden_size)
#         self.decoder_weight = self.create_parameter(
#             shape=[hidden_size, vocab_size],
#             dtype=self.transform.weight.dtype,
#             is_bias=True) if embedding_weights is None else embedding_weights
#         self.decoder_bias = self.create_parameter(
#             shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)
#
#     def forward(self, hidden_states, masked_positions=None):
#         if masked_positions is not None:
#             hidden_states = paddle.reshape(hidden_states,
#                                            [-1, hidden_states.shape[-1]])
#             hidden_states = paddle.tensor.gather(hidden_states,
#                                                  masked_positions)
#         # gather masked tokens might be more quick
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.layer_norm(hidden_states)
#         hidden_states = paddle.tensor.matmul(
#             hidden_states, self.decoder_weight,
#             transpose_y=True) + self.decoder_bias
#         return hidden_states
#
#
# class GPT2PretrainingHeads(nn.Layer):
#     def __init__(self,
#                  hidden_size,
#                  vocab_size,
#                  activation,
#                  embedding_weights=None):
#         super(GPT2PretrainingHeads, self).__init__()
#         self.predictions = GPT2LMPredictionHead(hidden_size, vocab_size,
#                                                 activation, embedding_weights)
#         # self.seq_relationship = nn.Linear(hidden_size, 2)
#
#     def forward(self, sequence_output, pooled_output, masked_positions=None):
#         prediction_scores = self.predictions(sequence_output, masked_positions)
#         return prediction_scores


class GPT2ForPretraining(GPT2PretrainedModel):
    def __init__(self, gpt2):
        super(GPT2ForPretraining, self).__init__()
        self.gpt2 = gpt2
        # self.cls = GPT2PretrainingHeads(
        #     self.bert.config["hidden_size"],
        #     self.bert.config["vocab_size"],
        #     self.bert.config["hidden_act"],
        #     embedding_weights=self.gpt2.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                kv_cache=None,
                use_cache=False):
        outputs = self.gpt2(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache)
        encoder_outputs, cached_kvs = outputs[:2]
        logits = paddle.matmul(
            encoder_outputs,
            self.gpt2.embeddings.word_embeddings.weight,
            transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPT2PretrainingCriterion(paddle.nn.Layer):
    def __init__(self, vocab_size):
        super(GPT2PretrainingCriterion, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, masked_lm_labels, masked_lm_scale):
        masked_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
            prediction_scores, masked_lm_labels, ignore_index=-1)
        masked_lm_loss = masked_lm_loss / masked_lm_scale
        return paddle.sum(masked_lm_loss)
