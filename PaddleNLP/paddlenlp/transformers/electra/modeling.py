# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F

from .. import PretrainedModel, register_base_model

__all__ = [
    'ElectraModel',
    'ElectraForTotalPretraining',
    'ElectraForPretraining',
    'ElectraForMaskedLM',
    'ElectraClassificationHead',
    'ElectraForSequenceClassification',
    'ElectraForTokenClassification',
    'ElectraModelOutput',
]


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(
            activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


class ElectraEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_dropout_prob,
                 max_position_embeddings,
                 type_vocab_size,
                 layer_norm_eps=1e-12):
        super(ElectraEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  embedding_size)

        self.layer_norm = nn.LayerNorm(embedding_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = paddle.arange(0, seq_length, dtype="int64")

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ElectraDiscriminatorPredictions(nn.Layer):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, hidden_size, hidden_act):
        super(ElectraDiscriminatorPredictions, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_prediction = nn.Linear(hidden_size, 1)
        self.act = get_activation(hidden_act)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.act(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()

        return logits


class ElectraGeneratorPredictions(nn.Layer):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, embedding_size, hidden_size, hidden_act):
        super(ElectraGeneratorPredictions, self).__init__()

        self.LayerNorm = nn.LayerNorm(embedding_size)
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.act = get_activation(hidden_act)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class ElectraPretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "electra"
    model_config_file = "model_config.json"

    # pretrained general configuration
    gen_weight = 1.0
    disc_weight = 50.0
    tie_word_embeddings = True
    untied_generator_embeddings = False
    untied_generator = True
    output_hidden_states = False
    output_attentions = False
    return_dict = False
    use_softmax_sample = True

    # model init configuration
    pretrained_init_configuration = {
        "electra-small-generator": {
            "architectures": ["ElectraForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "num_attention_heads": 4,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522
        },
        "electra-base-generator": {
            "architectures": ["ElectraForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "num_attention_heads": 4,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522
        },
        "electra-large-generator": {
            "architectures": ["ElectraForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 1024,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "num_attention_heads": 4,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522
        },
        "electra-small-discriminator": {
            "architectures": ["ElectraForPretraining"],
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "num_attention_heads": 4,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522
        },
        "electra-base-discriminator": {
            "architectures": ["ElectraForPretraining"],
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522
        },
        "electra-large-discriminator": {
            "architectures": ["ElectraForPretraining"],
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 1024,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522
        },
        "chinese-electra-discriminator-small": {
            "architectures": ["ElectraForPretraining"],
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 128,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 1024,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "num_attention_heads": 4,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 21128,
        },
        "chinese-electra-discriminator-base": {
            "architectures": ["ElectraForPretraining"],
            "attention_probs_dropout_prob": 0.1,
            "embedding_size": 768,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "electra",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 21128,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "electra-small-generator":
            "http://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-small-generator.pdparams",
            "electra-base-generator":
            "http://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-base-generator.pdparams",
            "electra-large-generator":
            "http://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-large-generator.pdparams",
            "electra-small-discriminator":
            "http://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-small-discriminator.pdparams",
            "electra-base-discriminator":
            "http://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-base-discriminator.pdparams",
            "electra-large-discriminator":
            "http://paddlenlp.bj.bcebos.com/models/transformers/electra/electra-large-discriminator.pdparamss",
            "chinese-electra-discriminator-small":
            "http://paddlenlp.bj.bcebos.com/models/transformers/chinese-electra-discriminator-small/chinese-electra-discriminator-small.pdparams",
            "chinese-electra-discriminator-base":
            "http://paddlenlp.bj.bcebos.com/models/transformers/chinese-electra-discriminator-base/chinese-electra-discriminator-base.pdparams",
        }
    }

    def init_weights(self):
        """
        Initializes and tie weights if needed.
        """
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights if needed
        self.tie_weights()

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if hasattr(self, "get_output_embeddings") and hasattr(
                self, "get_input_embeddings"):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings,
                                           self.get_input_embeddings())

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.electra.config["initializer_range"],
                    shape=module.weight.shape))
        elif isinstance(module, nn.LayerNorm):
            module.bias.set_value(paddle.zeros_like(module.bias))
            module.weight.set_value(paddle.full_like(module.weight, 1.0))
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.set_value(paddle.zeros_like(module.bias))

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights"""
        if output_embeddings.weight.shape == input_embeddings.weight.shape:
            output_embeddings.weight = input_embeddings.weight
        elif output_embeddings.weight.shape == input_embeddings.weight.t(
        ).shape:
            output_embeddings.weight.set_value(input_embeddings.weight.t())
        else:
            raise ValueError(
                "when tie input/output embeddings, the shape of output embeddings: {}"
                "should be equal to shape of input embeddings: {}"
                "or should be equal to the shape of transpose input embeddings: {}".
                format(output_embeddings.weight.shape, input_embeddings.weight.
                       shape, input_embeddings.weight.t().shape))
        if getattr(output_embeddings, "bias", None) is not None:
            if output_embeddings.weight.shape[
                    -1] != output_embeddings.bias.shape[0]:
                raise ValueError(
                    "the weight lase shape: {} of output_embeddings is not equal to the bias shape: {}"
                    "please check output_embeddings configuration".format(
                        output_embeddings.weight.shape[
                            -1], output_embeddings.bias.shape[0]))

    def get_extended_attention_mask(self, attention_mask, input_shape, place):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`paddle.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            place: (:obj:`paddle.Tensor.place`):
                The place of the input to the model.
        Returns:
            :obj:`paddle.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            #extended_attention_mask = attention_mask[:, None, :, :]
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".
                format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.to(dtype=self.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self,
                      head_mask,
                      num_hidden_layers,
                      is_attention_chunked=False):
        """
        Prepare the head mask if needed.
        Args:
            head_mask (:obj:`paddle.Tensor` with shape :obj:`[num_heads]` 
                or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            :obj:`paddle.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]
                or list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask,
                                                      num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1)
            head_mask = paddle.expand(head_mask,
                                      [num_hidden_layers, -1, -1, -1, -1])
        elif head_mask.dim() == 2:
            # We can specify head_mask for each layer
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert (head_mask.dim() == 5
                ), "head_mask.dim != 5, instead {head_mask.dim()}"
        #head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask


@dataclass
class ElectraModelOutput(OrderedDict):
    """
    Output type of :class:`ElectraPretrainedModel`.
    Args:
        loss (`optional`, returned when ``labels`` is provided, ``paddle.Tensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA objective.
        logits (:obj:`paddle.Tensor` dtype=float32 of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` dtype=float32 
            (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` dtype=float32 (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax,
            used to compute the weighted average in the self-attention heads.
    """
    loss = None
    logits = None
    hidden_states = None
    attentions = None


ELECTRA_START_DOCSTRING = r"""
    This model inherits from :class:`ElectraPretrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/dygraph/layers/Layer_cn.html#layer>`__
    subclass. Use it as a regular Paddle Module and refer to the Padddle documentation for all matter related to
    general usage and behavior.
    Parameters:
"""

ELECTRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`paddle.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (:obj:`paddle.Tensor` dtype=float32 of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        token_type_ids (:obj:`paddle.Tensor` dtype=int64 of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
        position_ids (:obj:`paddle.Tensor` dtype=int64 of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, max_position_embeddings - 1]``.
        head_mask (:obj:`paddle.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`paddle.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`paddle.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`paddle.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`ElectraModelOutput` instead of a plain tuple.
"""


@register_base_model
class ElectraModel(ElectraPretrainedModel):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 num_hidden_layers, num_attention_heads, intermediate_size,
                 hidden_act, hidden_dropout_prob, attention_probs_dropout_prob,
                 max_position_embeddings, type_vocab_size, layer_norm_eps,
                 pad_token_id, initializer_range, model_type, architectures):
        super(ElectraModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = ElectraEmbeddings(
            vocab_size, embedding_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, layer_norm_eps)

        if embedding_size != hidden_size:
            self.embeddings_project = nn.Linear(embedding_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        place = input_ids.place if input_ids is not None else inputs_embeds.place

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, place)
        #head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds)

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            extended_attention_mask
            #head_mask=head_mask,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
        )

        return (hidden_states, )


class ElectraForPretraining(ElectraPretrainedModel):
    def __init__(self, electra):
        super(ElectraForPretraining, self).__init__()

        self.electra = electra
        self.discriminator_predictions = ElectraDiscriminatorPredictions(
            self.electra.config["hidden_size"],
            self.electra.config["hidden_act"])
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (``paddle.Tensor`` dtype=ing64 of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        """
        return_dict = return_dict if return_dict is not None else self.return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict, )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = paddle.reshape(
                    attention_mask,
                    [-1, discriminator_sequence_output.shape[1]]) == 1
                active_logits = paddle.reshape(
                    logits,
                    [-1, discriminator_sequence_output.shape[1]])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.astype("float32"))
            else:
                loss = loss_fct(
                    paddle.reshape(
                        logits, [-1, discriminator_sequence_output.shape[1]]),
                    labels.astype("float32"))

        if not return_dict:
            output = (logits, ) + discriminator_hidden_states[1:]
            return ((loss, ) + output) if loss is not None else output

        return ElectraModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions, )


class ElectraForMaskedLM(ElectraPretrainedModel):
    def __init__(self, electra):
        super(ElectraForMaskedLM, self).__init__()

        self.electra = electra
        self.generator_predictions = ElectraGeneratorPredictions(
            self.electra.config["embedding_size"],
            self.electra.config["hidden_size"],
            self.electra.config["hidden_act"])

        if not self.tie_word_embeddings:
            self.generator_lm_head = nn.Linear(
                self.electra.config["embedding_size"],
                self.electra.config["vocab_size"])
        else:
            self.generator_lm_head_bias = paddle.fluid.layers.create_parameter(
                shape=[self.electra.config["vocab_size"]],
                dtype='float32',
                is_bias=True)
        self.init_weights()

    def get_input_embeddings(self):
        return self.electra.embeddings.word_embeddings

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        r"""
        labels (:obj:`paddle.Tensor` dtype = int64 of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        assert (kwargs == {}
                ), "Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.return_dict

        generator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict, )
        generator_sequence_output = generator_hidden_states[0]

        prediction_scores = self.generator_predictions(
            generator_sequence_output)
        if not self.tie_word_embeddings:
            prediction_scores = self.generator_lm_head(prediction_scores)
        else:
            prediction_scores = F.linear(
                prediction_scores,
                self.get_input_embeddings().weight.transpose([1, 0]),
                self.generator_lm_head_bias)

        loss = None
        # Masked language modeling softmax layer
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(
                reduction='none')  # -100 index = padding token
            loss = loss_fct(
                paddle.reshape(prediction_scores, [-1, self.vocab_size]),
                paddle.reshape(labels, [-1]))

            umask_positions = paddle.zeros_like(labels).astype("float32")
            mask_positions = paddle.ones_like(labels).astype("float32")
            mask_positions = paddle.where(labels == -100, umask_positions,
                                          mask_positions)
            loss = loss.sum() / mask_positions.sum()

        if not return_dict:
            output = (prediction_scores, ) + generator_hidden_states[1:]
            return ((loss, ) + output) if loss is not None else output

        return ElectraModelOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions, )


# class ElectraClassificationHead and ElectraForSequenceClassification for fine-tuning
class ElectraClassificationHead(nn.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super(ElectraClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraForSequenceClassification(ElectraPretrainedModel):
    def __init__(self, electra, num_labels):
        super(ElectraForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.electra = electra
        self.classifier = ElectraClassificationHead(
            self.electra.config["hidden_size"],
            self.electra.config["hidden_dropout_prob"], self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (:obj:`paddle.Tensor` dtype=int64 of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            num_labels - 1]`. If :obj:`num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict, )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)

        loss = None
        if not return_dict:
            output = (logits, ) + discriminator_hidden_states[1:]
            return ((loss, ) + output) if loss is not None else output

        return ElectraModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions, )


class ElectraForTokenClassification(ElectraPretrainedModel):
    def __init__(self, electra, num_labels):
        super(ElectraForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.electra = electra
        self.dropout = nn.Dropout(self.electra.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.electra.config["hidden_size"],
                                    self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (:obj:`paddle.Tensor` dtype=int64 of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. 
            Indices should be in ``[0, ..., num_labels-1]``.
        """
        return_dict = return_dict if return_dict is not None else self.return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict, )
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(
            discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)

        loss = None
        if not return_dict:
            output = (logits, ) + discriminator_hidden_states[1:]
            return ((loss, ) + output) if loss is not None else output

        return ElectraModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions, )


class ElectraForTotalPretraining(ElectraPretrainedModel):
    def __init__(self, generator, discriminator):
        super(ElectraForTotalPretraining, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.initializer_range = discriminator.electra.initializer_range
        self.init_weights()

    def get_input_embeddings(self):
        if not self.untied_generator_embeddings:
            return self.generator.electra.embeddings.word_embeddings
        else:
            return None

    def get_output_embeddings(self):
        if not self.untied_generator_embeddings:
            return self.discriminator.electra.embeddings.word_embeddings
        else:
            return None

    def get_discriminator_inputs(self, inputs, raw_inputs, mlm_logits, masked,
                                 use_softmax_sample):
        """Sample from the generator to create corrupted input."""
        # get generator token result
        sampled_tokens = (self.sample_from_softmax(mlm_logits,
                                                   use_softmax_sample)).detach()
        #sampled_tokens = self.sample_from_softmax(mlm_logits)
        sampled_tokids = paddle.argmax(sampled_tokens, axis=-1)
        # update token only at mask position
        # masked : [B, L], L contains -100(unmasked) or token value(masked)
        # mask_positions : [B, L], L contains 0(unmasked) or 1(masked)
        umask_positions = paddle.zeros_like(masked)
        mask_positions = paddle.ones_like(masked)
        mask_positions = paddle.where(masked == -100, umask_positions,
                                      mask_positions)
        updated_input = self.scatter_update(inputs, sampled_tokids,
                                            mask_positions)
        # use inputs and updated_input to generate labels
        labels = mask_positions * (paddle.ones_like(inputs) - paddle.equal(
            updated_input, raw_inputs).astype("int32"))
        return updated_input, labels, sampled_tokids

    def sample_from_softmax(self, logits, use_softmax_sample=True):
        if use_softmax_sample:
            #uniform_noise = paddle.uniform(logits.shape, dtype="float32", min=0, max=1)
            uniform_noise = paddle.rand(logits.shape, dtype="float32")
            gumbel_noise = -paddle.log(-paddle.log(uniform_noise + 1e-9) + 1e-9)
        else:
            gumbel_noise = paddle.zeros_like(logits)
        # softmax_sample equal to sampled_tokids.unsqueeze(-1)
        ins_softmax = nn.Softmax(axis=-1)
        softmax_sample = paddle.argmax(
            ins_softmax(logits + gumbel_noise), axis=-1)
        # one hot
        return F.one_hot(softmax_sample, logits.shape[-1])

    def scatter_update(self, sequence, updates, positions):
        """Scatter-update a sequence.
        Args:
          sequence: A [batch_size, seq_len] or [batch_size, seq_len, depth] tensor
          updates: A tensor of size batch_size*seq_len(*depth)
          positions: A [batch_size, n_positions] tensor
        Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
          [batch_size, seq_len, depth] tensor of "sequence" with elements at
          "positions" replaced by the values at "updates." Updates to index 0 are
          ignored. If there are duplicated positions the update is only applied once.
          Second is a [batch_size, seq_len] mask tensor of which inputs were updated.
        """
        shape = sequence.shape
        assert (len(shape) == 2), "the dimension of inputs should be [B, L]"
        B, L = shape
        N = positions.shape[1]
        assert (
            N == L), "the dimension of inputs and mask should be same as [B, L]"

        updated_sequence = ((
            (paddle.ones_like(sequence) - positions) * sequence) +
                            (positions * updates))

        return updated_sequence

    def forward(
            self,
            input_ids=None,
            raw_input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (``paddle.Tensor`` dtype=int64 of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:
            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.return_dict
        assert (
            labels is not None
        ), "labels should not be None, please check DataCollatorForLanguageModeling"

        generator_output = self.generator(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict, )
        loss = generator_output[0] * self.gen_weight
        logits = generator_output[1]

        discriminator_inputs, discriminator_labels, generator_predict_tokens = self.get_discriminator_inputs(
            input_ids, raw_input_ids, logits, labels, self.use_softmax_sample)

        discriminator_output = self.discriminator(
            discriminator_inputs,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            discriminator_labels,
            output_attentions,
            output_hidden_states,
            return_dict, )
        loss += discriminator_output[0] * self.disc_weight
        logits = discriminator_output[1]

        if not return_dict:
            return ((loss, ) + (logits, ))

        return ElectraModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=generator_output.hidden_states,
            attentions=generator_output.attentions, )
