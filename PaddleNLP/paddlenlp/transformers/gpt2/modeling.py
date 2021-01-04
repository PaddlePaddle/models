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
import collections
import numpy as np
import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
import math
from paddle.fluid import layers

from .. import PretrainedModel, register_base_model
from paddle.nn.layer.transformer import _convert_param_attr_to_list

__all__ = [
    'GPT2Model', "GPT2PretrainedModel", 'GPT2ForPretraining',
    'GPT2PretrainingCriterion'
]


def openai_glue(x):
    return 0.5 * x * (1.0 + paddle.tanh(0.7978845608028654 * x *
                                        (1.0 + 0.044715 * x * x)))


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.
        weight_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr` .

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = nn.Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = nn.Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`.
            value (Tensor): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`.
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiHeadAttention. If is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.

        Returns:
            tuple: A tuple including linear projected keys and values. These two \
                tensors have shapes `[batch_size, n_head, sequence_length, d_key]` \
                and `[batch_size, n_head, sequence_length, d_value]` separately, \
                and their data types are same as inputs.
        """

        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v) if use_cache is False else (q, k, v, cache)

    def compute_kv(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.

        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        Parameters:
            key (Tensor): The keys for multi-head attention. It is a tensor
                with shape `[batch_size, sequence_length, kdim]`. The data type
                should be float32 or float64.
            value (Tensor): The values for multi-head attention. It is a tensor
                with shape `[batch_size, sequence_length, vdim]`. The data type
                should be float32 or float64.

        Returns:
            tuple: A tuple including transformed keys and values. Their shapes \
                both are `[batch_size, num_heads, sequence_length, embed_dim // num_heads]`, \
                and their data types are same as inputs.
        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.

        `Cache` or `StaticCache` is namedtuple with `k` and `v` as fields,
        and it stores tensors shaped `[batch_size, num_heads, length, embed_dim]`
        which are results of linear projection, reshape and transpose calculations
        in MultiHeadAttention.

        If the generated cache is an instance of `Cache`, `k` and `v` fields
        reserve intermediate result tensors of previous positions, and the tensors
        are incremental among decoding steps, which mostly are used for decoder
        decoder self attention.

        If the generated cache is an instance of `StaticCache`, `k` and `v` fields
        would be used as calculated result tensors on keys an values in `forward`,
        and the tensors keep unchanged among decoding steps, which are mostly used
        for decoder-encoder cross attention.

        The cache is generated as follows:

        1. If `type` is `StaticCache`, apply `compute_kv(key, value)` and use the
        results to create an instance of `StaticCache`.

        2. If `type` is `Cache` and `value` is None, generate empty tensors shaped
        `[batch_size, num_heads, 0, embed_dim // num_heads]` and use the results
        to create an instance of `Cache`, where `batch_size` is from the first
        dimension of `key`.

        3. If `type` is `Cache` and `value` is not None, use `key`, `value` to create
        an instance of `Cache`.

        Parameters:
            key (Tensor): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If `value` is None,
                it is only for batch size and data type reference.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, `key` is only
                for batch size reference. Default None.
            type (type): It should be `MultiHeadAttention.StaticCache` or
                `MultiHeadAttention.Cache` to indicate the cache type to generate.

        Returns:
            namedtuple: an instance of `Cache` or `StaticCache` accordingly.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self,
                query,
                key,
                value,
                attn_mask=None,
                use_cache=False,
                cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiHeadAttention. If it is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if use_cache is False:
            q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache,
                                               cache)
        # scale dot product attention
        # TODO(guosheng): use tensor.matmul, however it doesn't support `alpha`
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)

        if attn_mask is not None:
            # TODO(guosheng): support bool mask
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache is True:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.

    Parameters:
        decoder_layer (Layer): an instance of the `TransformerDecoderLayer`. It
            would be used as the first layer, and the other layers would be created
            according to the configurations of it.
        num_layers (int): The number of decoder layers to be stacked.
        norm (LayerNorm, optional): the layer normalization component. If provided,
            apply layer normalization on the output of last encoder layer.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import TransformerDecoderLayer, TransformerDecoder

            # decoder input: [batch_size, tgt_len, d_model]
            dec_input = paddle.rand((2, 4, 128))
            # encoder output: [batch_size, src_len, d_model]
            enc_output = paddle.rand((2, 6, 128))
            # self attention mask: [batch_size, n_head, tgt_len, tgt_len]
            self_attn_mask = paddle.rand((2, 2, 4, 4))
            # cross attention mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 4, 6))
            decoder_layer = TransformerDecoderLayer(128, 2, 512)
            decoder = TransformerDecoder(decoder_layer, 2)
            output = decoder(dec_input,
                             enc_output,
                             self_attn_mask,
                             cross_attn_mask)  # [2, 4, 128]
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.LayerList([(
            decoder_layer
            if i == 0 else type(decoder_layer)(**decoder_layer._config))
                                    for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.checkpoints = []

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                use_cache=False,
                cache=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.

        Parameters:
            tgt (Tensor): The input of Transformer decoder. It is a tensor
                with shape `[batch_size, target_length, d_model]`. The data type
                should be float32 or float64.
            memory (Tensor): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            tgt_mask (Tensor, optional): A tensor used in self attention
                to prevents attention to some unwanted positions, usually the
                the subsequent positions. It is a tensor with shape broadcasted
                to `[batch_size, n_head, target_length, target_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None
            memory_mask (Tensor, optional): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions,
                usually the paddings. It is a tensor with shape broadcasted to
               `[batch_size, n_head, target_length, source_length]`, where the
                unwanted positions have `-INF` values and the others have 0 values.
                The data type should be float32 or float64. It can be None when
                nothing wanted or needed to be prevented attention to. Default None
            cache (list, optional): It is a list, and each element in the list
                is a tuple( :code:`(incremental_cache, static_cache)` ). See
                `TransformerDecoder.gen_cache` for more details. It is only
                used for inference and should be None for training. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `tgt`, representing the output of Transformer decoder. \
                Or a tuple if `cache` is not None, except for decoder output, \
                the tuple includes the new cache which is same as input `cache` \
                argument but `incremental_cache` in it has an incremental length. \
                See `MultiHeadAttention.gen_cache` and `MultiHeadAttention.forward` \
                for more details.
        """
        output = tgt
        new_caches = []
        self.checkpoints = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                if use_cache:
                    output, new_cache = mod(output,
                                            memory,
                                            tgt_mask=tgt_mask,
                                            use_cache=use_cache,
                                            cache=cache)
                    new_caches.append(new_cache)
                else:
                    output = mod(output,
                                 memory,
                                 tgt_mask=tgt_mask,
                                 use_cache=use_cache,
                                 cache=cache)

            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        use_cache=use_cache,
                                        cache=cache[i])
                new_caches.append(new_cache)
            self.checkpoints.append(output.name)

        if self.norm is not None:
            output = self.norm(output)
        return output if use_cache is False else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.


        Parameters:
            memory (Tensor): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            do_zip (bool, optional): Indicate whether to apply `zip` on the tuples.
                If True, return a list with two elements. Default False

        Returns:
            list: It is a list, and each element in the list is a tuple produced \
                by `TransformerDecoderLayer.gen_cache(memory)`. See `TransformerDecoderLayer.gen_cache` \
                for more details. If `do_zip` is True, apply `zip` on these tuples \
                and return a list with two elements.
        """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


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

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        #self.dropout1 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])
        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    use_cache, cache)
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        # tgt = self.dropout2(self.linear2(openai_glue(self.linear1(tgt))))
        tgt = self.dropout2(
            self.linear2(F.gelu(
                self.linear1(tgt), approximate=True)))
        tgt = residual + tgt

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        return tgt if use_cache is False else (tgt, incremental_cache)

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

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones

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
        "gpt2-base-cn": {
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
        "gpt2-medium-en": {
            "vocab_size": 50304,
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },
        "gpt2-small-en": {
            "vocab_size": 50304,
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "gpt2-base-cn": "http://10.255.129.12:8829/model_state.pdparams",
        }
    }
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
        self.decoder = TransformerDecoder(
            decoder_layer, num_hidden_layers, norm=nn.LayerNorm(hidden_size))
        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                cache=None):
        self.checkpoints = []
        if attention_mask is None:
            length = input_ids.shape[1]
            attention_mask = paddle.tensor.triu(
                (paddle.ones(
                    (length, length),
                    dtype=self.embeddings.word_embeddings.weight.dtype) * -1e9),
                1)
        if position_ids is None:
            # TODO(wawltor) for the static mode, must use the shape to calculate the shape
            past_length = 0
            if cache is not None:
                past_length = cache[0].k.shape[-2]
            position_ids = paddle.arange(
                past_length, input_ids.shape[-1] + past_length, dtype='int64')
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)
        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            use_cache=use_cache,
            cache=cache)
        self.checkpoints.extend(self.decoder.checkpoints)
        return encoder_outputs


class GPT2ForPretraining(GPT2PretrainedModel):
    def __init__(self, gpt2):
        super(GPT2ForPretraining, self).__init__()
        self.gpt2 = gpt2
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                use_cache=False,
                cache=None):
        outputs = self.gpt2(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = paddle.matmul(
            encoder_outputs,
            self.gpt2.embeddings.word_embeddings.weight,
            transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPT2PretrainingCriterion(paddle.nn.Layer):
    def __init__(self):
        super(GPT2PretrainingCriterion, self).__init__()
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        masked_lm_loss = self.loss_func(prediction_scores,
                                        masked_lm_labels.unsqueeze(2))
        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        loss = masked_lm_loss / loss_mask.sum()
        return loss
