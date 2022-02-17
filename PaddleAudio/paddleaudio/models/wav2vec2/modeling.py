# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
""" Wav2Vec2 model """

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any, Optional, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.utils import download

from .activations import ACT2FN
from .config_utils import PretrainedConfig

__all__ = ['Wav2Vec2ForCTC', 'Wav2Vec2Model']

CONFIG_NAMES = [
    'wav2vec2-base-960h',
    'wav2vec2-large-960h',
    'wav2vec2-large-960h-lv60',
    'wav2vec2-large-960h-lv60-self',
]
URL_BASE = 'https://bj.bcebos.com/paddleaudio/models/wav2vec2/'
AUDIO_URL = 'https://bj.bcebos.com/paddleaudio/test/data/librispeech/sample1.flac'
TEXT_URL = 'https://bj.bcebos.com/paddleaudio/test/data/librispeech/sample1.txt'


def is_tensor(x):
    return isinstance(x, paddle.Tensor)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """
    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (not isinstance(element,
                                       (list, tuple)) or not len(element) == 2
                            or not isinstance(element[0], str)):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutput(ModelOutput):

    last_hidden_state: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class CausalLMOutput(BaseModelOutput):

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


@dataclass
class MaskedLMOutput(BaseModelOutput):

    loss: Optional[paddle.Tensor] = None
    logits: paddle.Tensor = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
]


def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[paddle.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        attention_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    Adapted from `fairseq's data_utils.py
    <https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376>`__.
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length) + np.random.rand())

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([
            mask_idc[j] + offset for j in range(len(mask_idc))
            for offset in range(lengths[j])
        ])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


class Wav2Vec2NoLayerNormConvLayer(nn.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1D(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias_attr=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2GroupNormConvLayer(nn.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1D(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias_attr=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim,
                                       num_channels=self.out_conv_dim)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        # paddle's groupnorm only supports 4D tensor as of 2.1.1. We need to unsqueeze and squeeze.
        hidden_states = self.layer_norm(hidden_states.unsqueeze([-1]))
        hidden_states = hidden_states[:, :, :, 0]
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2PositionalConvEmbedding(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1D(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        # self.conv = nn.utils.weight_norm(self.conv, name="weight", axis=2)
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose((0, 2, 1))

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose((0, 2, 1))
        return hidden_states


class Wav2Vec2SamePadLayer(nn.Layer):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, :-self.num_pad_remove]
        return hidden_states


class Wav2Vec2FeatureExtractor(nn.Layer):
    """Construct the featurs from raw audio waveform"""
    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i)
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.LayerList(conv_layers)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input_values):
        #hidden_states = input_values[:, None]
        hidden_states = input_values.unsqueeze(1)

        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = nn.Conv1D(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias_attr=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(self.out_conv_dim,
                                       weight_attr=True,
                                       bias_attr=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = hidden_states.transpose((0, 2, 1))
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose((0, 2, 1))

        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2FeatureProjection(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1],
                                       epsilon=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Wav2Vec2Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape(
            (bsz, seq_len, self.num_heads, self.head_dim)).transpose(
                (0, 2, 1, 3))  #?.contiguous()

    def forward(
        self,
        hidden_states: paddle.Tensor,
        key_value_states: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        layer_head_mask: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor],
               Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states],
                                         axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(paddle.Tensor, paddle.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(paddle.Tensor, paddle.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len,
                                   bsz).reshape(proj_shape)
        key_states = key_states.reshape(proj_shape)
        value_states = value_states.reshape(proj_shape)

        src_len = key_states.shape[1]
        attn_weights = paddle.bmm(query_states, key_states.transpose((0, 2, 1)))

        assert attn_weights.shape == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ], f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.shape}"

        if attention_mask is not None:
            assert attention_mask.shape == [
                bsz,
                1,
                tgt_len,
                src_len,
            ], f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
            attn_weights = attn_weights.reshape(
                (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = attn_weights.reshape(
                (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = F.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.reshape(
                (1, -1, 1, 1)) * attn_weights.reshape(
                    (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = attn_weights.reshape(
                (bsz * self.num_heads, tgt_len, src_len))

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.reshape(
                (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = attn_weights_reshaped.reshape(
                (bsz * self.num_heads, tgt_len, src_len))
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights,
                               p=self.dropout,
                               training=self.training)

        attn_output = paddle.bmm(attn_probs, value_states)

        assert attn_output.shape == [
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ], f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (attn_output.reshape(
            (bsz, self.num_heads, tgt_len, self.head_dim)).transpose(
                (0, 2, 1, 3)).reshape((bsz, tgt_len, embed_dim)))

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class Wav2Vec2FeedForward(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size,
                                            config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size,
                                      config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2Output(nn.Layer):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, input_tensor):
        return hidden_states


class Wav2Vec2EncoderLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       epsilon=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             epsilon=config.layer_norm_eps)

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (attn_weights, )

        return outputs


class Wav2Vec2Encoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.LayerList([
            Wav2Vec2EncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        assert output_hidden_states is False
        assert output_attentions is False
        # all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(
                dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(attention_mask.shape[0], 1,
                                                   attention_mask.shape[-1],
                                                   attention_mask.shape[-1])
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            #  if output_hidden_states:
            #all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            layer_outputs = layer(hidden_states,
                                  attention_mask=attention_mask,
                                  output_attentions=output_attentions)
            hidden_states = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )


class Wav2Vec2PretrainedModel(nn.Layer):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "wav2vec2"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1D):
            paddle.nn.initializer.KaimingNormal(module.weight.data)
        if isinstance(module,
                      (nn.Linear, nn.Conv1D)) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self, ):
        self.apply(self._init_weights)

    def _get_feat_extract_output_lengths(self, input_lengths: paddle.Tensor):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1D.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel,
                                       self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths.astype('int64')


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       epsilon=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             epsilon=config.layer_norm_eps)

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions)
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(
            self.final_layer_norm(hidden_states))

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (attn_weights, )

        return outputs


class Wav2Vec2EncoderStableLayerNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.LayerList([
            Wav2Vec2EncoderLayerStableLayerNorm(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states[~attention_mask] = 0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(
                dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(attention_mask.shape[0], 1,
                                                   attention_mask.shape[-1],
                                                   attention_mask.shape[-1])

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.config.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = layer(hidden_states,
                                      attention_mask=attention_mask,
                                      output_attentions=output_attentions)
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1], )

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v for v in
                [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Wav2Vec2Model(Wav2Vec2PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        self.masked_spec_embed = paddle.create_parameter((config.hidden_size, ),
                                                         'float32')

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = False  #output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = False  # (
        # output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        #)
        return_dict = True  # return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose((0, 2, 1))

        if attention_mask is not None:
            # compute real output lengths according to convolution formula
            output_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1))

            attention_mask = paddle.zeros(hidden_states.shape[:2],
                                          dtype=hidden_states.dtype)
            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            attention_mask[(paddle.arange(0, end=attention_mask.shape[0]),
                            output_lengths - 1)] = 1
            attention_mask = attention_mask.flip([-1
                                                  ]).cumsum(-1).flip([-1
                                                                      ]).bool()

        hidden_states = self.feature_projection(hidden_states)

        if self.config.apply_spec_augment and self.training:
            batch_size, sequence_length, hidden_size = hidden_states.shape
            assert False
            # apply SpecAugment along time axis
            if self.config.mask_time_prob > 0:
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    self.config.mask_time_prob,
                    self.config.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=2,
                )
                hidden_states[paddle.to_tensor(
                    mask_time_indices)] = self.masked_spec_embed.to(
                        hidden_states.dtype)

            # apply SpecAugment along feature axis
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = paddle.to_tensor(mask_feature_indices)
                hidden_states[mask_feature_indices[:, None].expand(
                    -1, sequence_length, -1)] = 0
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Wav2Vec2ForCTC(Wav2Vec2PretrainedModel):
    def __init__(self, config_name, pretrained=True):
        super().__init__()

        assert config_name in CONFIG_NAMES, f'input config {config_name} incorrect, available configs: {CONFIG_NAMES}'

        config_url = URL_BASE + f'config/{config_name}.json'
        config_path = download.get_weights_path_from_url(config_url)
        config = PretrainedConfig.from_pretrained(config_path)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.config = config
        if pretrained:
            weight_url = URL_BASE + f'weights/{config_name}.pdparam'
            weight_path = download.get_weights_path_from_url(weight_url)
            state_dict = paddle.load(weight_path)
            self.load_dict(state_dict)
            self.eval()

    @classmethod
    def from_pretrained(cls, config_name):

        assert config_name in CONFIG_NAMES, f'input config {config_name} incorrect, available configs: {CONFIG_NAMES}'
        weight_url = URL_BASE + f'weights/{config_name}.pdparam'
        weight_path = download.get_weights_path_from_url(weight_url)

        config_url = URL_BASE + f'config/{config_name}.json'
        config_path = download.get_weights_path_from_url(config_url)

        config = PretrainedConfig.from_pretrained(config_path)
        model = cls(config)
        state_dict = paddle.load(weight_path)
        model.load_dict(state_dict)
        model.eval()
        return model

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        labels=None,
    ):
        assert return_dict is True
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits
