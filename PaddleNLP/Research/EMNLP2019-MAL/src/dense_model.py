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
# limitations under the License

from functools import partial
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layer_helper import LayerHelper as LayerHelper

from config import *
from beam_search import BeamSearch

INF = 1. * 1e5

def layer_norm(x, begin_norm_axis=1, epsilon=1e-6, param_attr=None, bias_attr=None):
    """
        layer_norm
    """
    helper = LayerHelper('layer_norm', **locals())
    mean = layers.reduce_mean(x, dim=begin_norm_axis, keep_dim=True)
    shift_x = layers.elementwise_sub(x=x, y=mean, axis=0)
    variance = layers.reduce_mean(layers.square(shift_x), dim=begin_norm_axis, keep_dim=True)
    r_stdev = layers.rsqrt(variance + epsilon)
    norm_x = layers.elementwise_mul(x=shift_x, y=r_stdev, axis=0)

    param_shape = [reduce(lambda x, y: x * y, norm_x.shape[begin_norm_axis:])]
    param_dtype = norm_x.dtype
    scale = helper.create_parameter(
        attr=param_attr,
        shape=param_shape,
        dtype=param_dtype,
        default_initializer=fluid.initializer.Constant(1.))
    bias = helper.create_parameter(
        attr=bias_attr,
        shape=param_shape,
        dtype=param_dtype,
        is_bias=True,
        default_initializer=fluid.initializer.Constant(0.))

    out = layers.elementwise_mul(x=norm_x, y=scale, axis=-1)
    out = layers.elementwise_add(x=out, y=bias, axis=-1)

    return out

def dense_position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(np.arange(
        num_timescales) * -log_timescale_increment)
        #num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         cache=None,
                         attention_type="dot_product",):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activiation to mask certain selected positions so that
    they will not considered in attention weights.
    """
    keys = queries if keys is None else keys
    values = keys if values is None else values

    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs: quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        """
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * n_head,
                      bias_attr=False,
                      num_flatten_dims=2)
        k = layers.fc(input=keys,
                      size=d_key * n_head,
                      bias_attr=False,
                      num_flatten_dims=2)
        v = layers.fc(input=values,
                      size=d_value * n_head,
                      bias_attr=False,
                      num_flatten_dims=2)
        return q, k, v

    def __split_heads(x, n_head):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] then output a tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        if n_head == 1:
            return x

        hidden_size = x.shape[-1]
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        reshaped = layers.reshape(
            x=x, shape=[0, 0, n_head, hidden_size // n_head], inplace=True)

        # permuate the dimensions into:
        # [batch_size, n_head, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to __split_heads.
        """
        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        # The value 0 in shape attr means copying the corresponding dimension
        # size of the input as the output dimension size.
        return layers.reshape(
            x=trans_x,
            shape=[0, 0, trans_x.shape[2] * trans_x.shape[3]],
            inplace=True)

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """
        scaled_q = layers.scale(x=q, scale=d_key ** -0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                seed=DenseModelHyperParams.dropout_seed,
                is_test=False, dropout_implementation='upscale_in_train')
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = layers.concat([cache['k'], k], axis=1)
        v = layers.concat([cache['v'], v], axis=1)
        layers.assign(k, cache['k'])
        layers.assign(v, cache['v'])

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key, #d_model,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         bias_attr=False,
                         num_flatten_dims=2)
    return proj_out


def positionwise_feed_forward(x, d_inner_hid, d_hid, dropout_rate):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       num_flatten_dims=2,
                       act="relu")
    if dropout_rate:
        hidden = layers.dropout(
            hidden,
            dropout_prob=dropout_rate,
            seed=DenseModelHyperParams.dropout_seed,
            is_test=False, dropout_implementation='upscale_in_train')
    out = layers.fc(input=hidden, size=d_hid, num_flatten_dims=2)
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout_rate=0.):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":  # add residual connection
            out = out + prev_out if prev_out else out
        elif cmd == "n":  # add layer normalization
            out = layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                epsilon=1e-6,
                param_attr=fluid.initializer.Constant(1.),
                bias_attr=fluid.initializer.Constant(0.))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    seed=DenseModelHyperParams.dropout_seed,
                    is_test=False, dropout_implementation='upscale_in_train')
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def prepare_encoder_decoder(src_word,
                            src_pos,
                            src_vocab_size,
                            src_emb_dim,
                            src_max_len,
                            dropout_rate=0.,
                            word_emb_param_name=None,
                            training=True,
                            pos_enc_param_name=None, 
                            is_src=True,
                            params_type="normal"):
    """Add word embeddings and position encodings.
    The output tensor has a shape of:
    [batch_size, max_src_length_in_batch, d_model].
    This module is used at the bottom of the encoder stacks.
    """
    assert params_type == "fixed" or params_type == "normal" or params_type == "new"
    pre_name = "densedense"

    if params_type == "fixed":
        pre_name = "fixed_densefixed_dense"
    elif params_type == "new":
        pre_name = "new_densenew_dense"

    src_word_emb = layers.embedding(
        src_word,
        size=[src_vocab_size, src_emb_dim],
        padding_idx=DenseModelHyperParams.bos_idx,  # set embedding of bos to 0
        param_attr=fluid.ParamAttr(
            name = pre_name + word_emb_param_name,
            initializer=fluid.initializer.Normal(0., src_emb_dim ** -0.5)))#, is_sparse=True)
    if not is_src and training:
        src_word_emb = layers.pad(src_word_emb, [0, 0, 1, 0, 0, 0])
    src_word_emb = layers.scale(x=src_word_emb, scale=src_emb_dim ** 0.5)
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            trainable=False, name = pre_name + pos_enc_param_name))
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc
    return layers.dropout(
        enc_input,
        dropout_prob=dropout_rate,
        seed=DenseModelHyperParams.dropout_seed,
        is_test=False, dropout_implementation='upscale_in_train') if dropout_rate else enc_input


prepare_encoder = partial(
    prepare_encoder_decoder, pos_enc_param_name="src_pos_enc_table", is_src=True)
prepare_decoder = partial(
    prepare_encoder_decoder, pos_enc_param_name="trg_pos_enc_table", is_src=False)


def encoder_layer(enc_input,
                  attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  preprocess_cmd="n",
                  postprocess_cmd="da"):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    attn_output = multi_head_attention(
        pre_process_layer(enc_input, preprocess_cmd,
                          prepostprocess_dropout), None, None, attn_bias, d_key,
        d_value, d_model, n_head, attention_dropout)
    attn_output = post_process_layer(enc_input, attn_output, postprocess_cmd,
                                     prepostprocess_dropout)
    ffd_output = positionwise_feed_forward(
        pre_process_layer(attn_output, preprocess_cmd, prepostprocess_dropout),
        d_inner_hid, d_model, relu_dropout)
    return post_process_layer(attn_output, ffd_output, postprocess_cmd,
                              prepostprocess_dropout)


def encoder(enc_input,
            attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd="n",
            postprocess_cmd="da"):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    stack_layer_norm = []
    bottom_embedding_output = pre_process_layer(enc_input, preprocess_cmd, prepostprocess_dropout)
    stack_layer_norm.append(bottom_embedding_output)

    #zeros = layers.zeros_like(enc_input)
    #ones_flag = layers.equal(zeros, zeros)
    #ones = layers.cast(ones_flag, 'float32')

    for i in range(n_layer):
        enc_output = encoder_layer(
            enc_input,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd, )
        enc_output_2 = pre_process_layer(enc_output, preprocess_cmd, prepostprocess_dropout)
        stack_layer_norm.append(enc_output_2)
         
        pre_output = bottom_embedding_output
        for index in xrange(1, len(stack_layer_norm)):
            pre_output = pre_output + stack_layer_norm[index]

        # pre_mean
        enc_input = pre_output / len(stack_layer_norm)

    enc_output = pre_process_layer(enc_output, preprocess_cmd,
                                   prepostprocess_dropout)
    return enc_output


def decoder_layer(dec_input,
                  enc_output,
                  slf_attn_bias,
                  dec_enc_attn_bias,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  preprocess_cmd,
                  postprocess_cmd,
                  cache=None):
    """ The layer to be stacked in decoder part.
    The structure of this module is similar to that in the encoder part except
    a multi-head attention is added to implement encoder-decoder attention.
    """
    slf_attn_output = multi_head_attention(
        pre_process_layer(dec_input, preprocess_cmd, prepostprocess_dropout),
        None,
        None,
        slf_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        cache, )
    slf_attn_output = post_process_layer(
        dec_input,
        slf_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    enc_attn_output = multi_head_attention(
        pre_process_layer(slf_attn_output, preprocess_cmd,
                          prepostprocess_dropout),
        enc_output,
        enc_output,
        dec_enc_attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout, )
    enc_attn_output = post_process_layer(
        slf_attn_output,
        enc_attn_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    ffd_output = positionwise_feed_forward(
        pre_process_layer(enc_attn_output, preprocess_cmd,
                          prepostprocess_dropout),
        d_inner_hid,
        d_model,
        relu_dropout, )
    dec_output = post_process_layer(
        enc_attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout, )
    return dec_output


def decoder(dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
            n_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
            caches=None):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    """
    for i in range(n_layer):
        dec_output = decoder_layer(
            dec_input,
            enc_output,
            dec_slf_attn_bias,
            dec_enc_attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            preprocess_cmd,
            postprocess_cmd,
            cache=None if caches is None else caches[i])
        dec_input = dec_output
    dec_output = pre_process_layer(dec_output, preprocess_cmd,
                                   prepostprocess_dropout)
    return dec_output


def make_all_inputs(input_fields):
    """
    Define the input data layers for the transformer model.
    """
    inputs = []
    for input_field in input_fields:
        input_var = layers.data(
            name=input_field,
            shape=input_descs[input_field][0],
            dtype=input_descs[input_field][1],
            lod_level=input_descs[input_field][2]
            if len(input_descs[input_field]) == 3 else 0,
            append_batch_size=False)
        inputs.append(input_var)
    return inputs

def make_all_py_reader_inputs(input_fields, is_test=False):
    """
    Define the input data layers for the transformer model.
    """
    reader = layers.py_reader(
        capacity=20,
        name="test_reader" if is_test else "train_reader",
        shapes=[dense_input_descs[input_field][0] for input_field in input_fields],
        dtypes=[dense_input_descs[input_field][1] for input_field in input_fields],
        lod_levels=[
            dense_input_descs[input_field][2]
            if len(dense_input_descs[input_field]) == 3 else 0
            for input_field in input_fields
        ], use_double_buffer=True)
    return layers.read_file(reader), reader


def dense_transformer(src_vocab_size,
                trg_vocab_size,
                max_length,
                n_layer,
                enc_n_layer,
                n_head,
                d_key,
                d_value,
                d_model,
                d_inner_hid,
                prepostprocess_dropout,
                attention_dropout,
                relu_dropout,
                preprocess_cmd,
                postprocess_cmd,
                weight_sharing,
                embedding_sharing,
                label_smooth_eps,
                use_py_reader=False,
                is_test=False,
                params_type="normal",
                all_data_inputs=None):
    """
        transformer
    """
    if embedding_sharing:
        assert src_vocab_size == trg_vocab_size, (
            "Vocabularies in source and target should be same for weight sharing."
        )


    data_input_names = encoder_data_input_fields + \
                decoder_data_input_fields[:-1] + label_data_input_fields + dense_bias_input_fields

    if use_py_reader:
        all_inputs = all_data_inputs
    else:
        all_inputs = make_all_inputs(data_input_names)

    enc_inputs_len = len(encoder_data_input_fields)
    dec_inputs_len = len(decoder_data_input_fields[:-1])
    enc_inputs = all_inputs[0:enc_inputs_len]
    dec_inputs = all_inputs[enc_inputs_len:enc_inputs_len + dec_inputs_len]
    real_label = all_inputs[enc_inputs_len + dec_inputs_len]
    weights = all_inputs[enc_inputs_len + dec_inputs_len + 1]
    reverse_label = all_inputs[enc_inputs_len + dec_inputs_len + 2]
    enc_inputs[2] = all_inputs[-3] # dense_src_slf_attn_bias
    dec_inputs[3] = all_inputs[-2] # dense_trg_slf_attn_bias
    dec_inputs[4] = all_inputs[-1] # dense_trg_src_attn_bias

    enc_output = wrap_encoder(
        src_vocab_size,
        max_length,
        enc_n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        embedding_sharing,
        enc_inputs,
        params_type=params_type)

    predict = wrap_decoder(
        trg_vocab_size,
        max_length,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        embedding_sharing,
        dec_inputs,
        enc_output, is_train = True if not is_test else False,
        params_type=params_type)

    # Padding index do not contribute to the total loss. The weights is used to
    # cancel padding index in calculating the loss.
    if label_smooth_eps:
        label = layers.one_hot(input=real_label, depth=trg_vocab_size)
        label = label * (1 - label_smooth_eps) + (1 - label) * (
            label_smooth_eps / (trg_vocab_size - 1))
        label.stop_gradient = True
    else:
        label = real_label

    cost = layers.softmax_with_cross_entropy(
        logits=predict,
        label=label,
        soft_label=True if label_smooth_eps else False)
    weighted_cost = cost * weights
    sum_cost = layers.reduce_sum(weighted_cost)
    sum_cost.persistable = True
    token_num = layers.reduce_sum(weights)
    token_num.persistable = True
    token_num.stop_gradient = True
    avg_cost = sum_cost / token_num

    sen_count = layers.shape(dec_inputs[0])[0]
    batch_predict = layers.reshape(predict, shape = [sen_count, -1, DenseModelHyperParams.trg_vocab_size])
    batch_label = layers.reshape(real_label, shape=[sen_count, -1])
    batch_weights = layers.reshape(weights, shape=[sen_count, -1, 1])
    return sum_cost, avg_cost, token_num, batch_predict, cost, sum_cost, batch_label, batch_weights


def wrap_encoder(src_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 embedding_sharing,
                 enc_inputs=None,
                 params_type="normal"):
    """
    The wrapper assembles together all needed layers for the encoder.
    """
    if enc_inputs is None:
        # This is used to implement independent encoder program in inference.
        src_word, src_pos, src_slf_attn_bias = make_all_inputs(
            encoder_data_input_fields)
    else:
        src_word, src_pos, src_slf_attn_bias = enc_inputs
    enc_input = prepare_encoder(
        src_word,
        src_pos,
        src_vocab_size,
        d_model,
        max_length,
        prepostprocess_dropout,
        word_emb_param_name=dense_word_emb_param_names[0],
        params_type=params_type)
    enc_output = encoder(
        enc_input,
        src_slf_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd, )
    return enc_output


def wrap_decoder(trg_vocab_size,
                 max_length,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 preprocess_cmd,
                 postprocess_cmd,
                 weight_sharing,
                 embedding_sharing,
                 dec_inputs=None,
                 enc_output=None,
                 caches=None, is_train=True, params_type="normal"):
    """
    The wrapper assembles together all needed layers for the decoder.
    """
    if dec_inputs is None:
        # This is used to implement independent decoder program in inference.
        trg_word, reverse_trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, enc_output = \
            make_all_inputs(dense_decoder_data_input_fields)
    else:
        trg_word, reverse_trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias = dec_inputs

    dec_input = prepare_decoder(
        trg_word,
        trg_pos,
        trg_vocab_size,
        d_model,
        max_length,
        prepostprocess_dropout,
        word_emb_param_name=dense_word_emb_param_names[0]
        if embedding_sharing else dense_word_emb_param_names[1], 
        training=is_train,
        params_type=params_type)

    dec_output = decoder(
        dec_input,
        enc_output,
        trg_slf_attn_bias,
        trg_src_attn_bias,
        n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        caches=caches)
    # Reshape to 2D tensor to use GEMM instead of BatchedGEMM
    dec_output = layers.reshape(
        dec_output, shape=[-1, dec_output.shape[-1]], inplace=True)

    assert params_type == "fixed" or params_type == "normal" or params_type == "new"
    pre_name = "densedense"
    if params_type == "fixed":
        pre_name = "fixed_densefixed_dense"
    elif params_type == "new":
        pre_name = "new_densenew_dense"
    if weight_sharing and embedding_sharing:
        predict = layers.matmul(
            x=dec_output,
            y=fluid.default_main_program().global_block().var(
                pre_name + dense_word_emb_param_names[0]),
            transpose_y=True)
    elif weight_sharing:
        predict = layers.matmul(
            x=dec_output,
            y=fluid.default_main_program().global_block().var(
                pre_name +  dense_word_emb_param_names[1]),
            transpose_y=True)
    else:
        predict = layers.fc(input=dec_output,
                            size=trg_vocab_size,
                            bias_attr=False)
    #layers.Print(predict, message="logits", summarize=20)
    if dec_inputs is None:
        # Return probs for independent decoder program.
        predict = layers.softmax(predict)
    return predict

    
def get_enc_bias(source_inputs):
    """
        get_enc_bias
    """
    source_inputs = layers.cast(source_inputs, 'float32')
    emb_sum = layers.reduce_sum(layers.abs(source_inputs), dim=-1)
    zero = layers.fill_constant([1], 'float32', value=0) 
    bias = layers.cast(layers.equal(emb_sum, zero), 'float32') * -1e9
    return layers.unsqueeze(layers.unsqueeze(bias, axes=[1]), axes=[1])


def dense_fast_decode(
        src_vocab_size,
        trg_vocab_size,
        max_in_len,
        n_layer,
        enc_n_layer,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        preprocess_cmd,
        postprocess_cmd,
        weight_sharing,
        embedding_sharing,
        beam_size,
        batch_size, 
        max_out_len,
        decode_alpha,
        eos_idx,
        params_type="normal"):
    """
    Use beam search to decode. Caches will be used to store states of history
    steps which can make the decoding faster.
    """

    assert params_type == "normal" or params_type == "new" or params_type == "fixed"
    data_input_names = dense_encoder_data_input_fields + fast_decoder_data_input_fields

    all_inputs = make_all_inputs(data_input_names)

    enc_inputs_len = len(encoder_data_input_fields)
    dec_inputs_len = len(fast_decoder_data_input_fields)
    enc_inputs = all_inputs[0:enc_inputs_len]
    dec_inputs = all_inputs[enc_inputs_len:enc_inputs_len + dec_inputs_len]

    enc_output = wrap_encoder(src_vocab_size, max_in_len, enc_n_layer, n_head,
                              d_key, d_value, d_model, d_inner_hid,
                              prepostprocess_dropout, attention_dropout,
                              relu_dropout, preprocess_cmd, postprocess_cmd,
                              weight_sharing, embedding_sharing, enc_inputs, params_type=params_type)
    enc_bias = get_enc_bias(enc_inputs[0])
    source_length, = dec_inputs

    def beam_search(enc_output, enc_bias, source_length):
        """
            beam_search
        """
        max_len = layers.fill_constant(
            shape=[1], dtype='int64', value=max_out_len)
        step_idx = layers.fill_constant(
            shape=[1], dtype='int64', value=0)
        cond = layers.less_than(x=step_idx, y=max_len)
        while_op = layers.While(cond)

        caches_batch_size = batch_size * beam_size
        init_score = np.zeros([1, beam_size]).astype('float32')
        init_score[:, 1:] = -INF
        initial_log_probs = layers.assign(init_score)

        alive_log_probs = layers.expand(initial_log_probs, [batch_size, 1])
        # alive seq [batch_size, beam_size, 1]
        initial_ids = layers.zeros([batch_size, 1, 1], 'float32')
        alive_seq = layers.expand(initial_ids, [1, beam_size, 1]) 
        alive_seq = layers.cast(alive_seq, 'int64')

        enc_output = layers.unsqueeze(enc_output, axes=[1])
        enc_output = layers.expand(enc_output, [1, beam_size, 1, 1])
        enc_output = layers.reshape(enc_output, [caches_batch_size, -1, d_model])

        tgt_src_attn_bias = layers.unsqueeze(enc_bias, axes=[1])
        tgt_src_attn_bias = layers.expand(tgt_src_attn_bias, [1, beam_size, n_head, 1, 1]) 
        enc_bias_shape = layers.shape(tgt_src_attn_bias)
        tgt_src_attn_bias = layers.reshape(tgt_src_attn_bias, [-1, enc_bias_shape[2], 
                enc_bias_shape[3], enc_bias_shape[4]])

        beam_search = BeamSearch(beam_size, batch_size, decode_alpha, trg_vocab_size, d_model)

        caches = [{
            "k": layers.fill_constant(
                shape=[caches_batch_size, 0, d_model],
                dtype=enc_output.dtype,
                value=0),
            "v": layers.fill_constant(
                shape=[caches_batch_size, 0, d_model],
                dtype=enc_output.dtype,
                value=0)
        } for i in range(n_layer)]

        finished_seq = layers.zeros_like(alive_seq)
        finished_scores = layers.fill_constant([batch_size, beam_size], 
                                                dtype='float32', value=-INF)
        finished_flags = layers.fill_constant([batch_size, beam_size], 
                                                dtype='float32', value=0)

        with while_op.block():
            pos = layers.fill_constant([caches_batch_size, 1, 1], dtype='int64', value=1)
            pos = layers.elementwise_mul(pos, step_idx, axis=0)

            alive_seq_1 = layers.reshape(alive_seq, [caches_batch_size, -1])
            alive_seq_2 = alive_seq_1[:, -1:] 
            alive_seq_2 = layers.unsqueeze(alive_seq_2, axes=[1])

            logits = wrap_decoder(
                trg_vocab_size, max_in_len, n_layer, n_head, d_key,
                d_value, d_model, d_inner_hid, prepostprocess_dropout,
                attention_dropout, relu_dropout, preprocess_cmd,
                postprocess_cmd, weight_sharing, embedding_sharing,
                dec_inputs=(alive_seq_2, alive_seq_2, pos, None, tgt_src_attn_bias),
                enc_output=enc_output, caches=caches, is_train=False, params_type=params_type)

            alive_seq_2, alive_log_probs_2, finished_seq_2, finished_scores_2, finished_flags_2, caches_2 = \
                    beam_search.inner_func(step_idx, logits, alive_seq_1, alive_log_probs, finished_seq, 
                                           finished_scores, finished_flags, caches, enc_output, 
                                           tgt_src_attn_bias)

            layers.increment(x=step_idx, value=1.0, in_place=True)
            finish_cond = beam_search.is_finished(step_idx, source_length, alive_log_probs_2, 
                                                  finished_scores_2, finished_flags_2) 

            layers.assign(alive_seq_2, alive_seq)
            layers.assign(alive_log_probs_2, alive_log_probs)
            layers.assign(finished_seq_2, finished_seq)
            layers.assign(finished_scores_2, finished_scores)
            layers.assign(finished_flags_2, finished_flags)

            for i in xrange(len(caches_2)):
                layers.assign(caches_2[i]["k"], caches[i]["k"])
                layers.assign(caches_2[i]["v"], caches[i]["v"])

            layers.logical_and(x=cond, y=finish_cond, out=cond)

        finished_flags = layers.reduce_sum(finished_flags, dim=1, keep_dim=True) / beam_size
        finished_flags = layers.cast(finished_flags, 'bool')
        mask = layers.cast(layers.reduce_any(input=finished_flags, dim=1, keep_dim=True), 'float32')
        mask = layers.expand(mask, [1, beam_size])

        mask2 = 1.0 - mask
        finished_seq = layers.cast(finished_seq, 'float32')
        alive_seq = layers.cast(alive_seq, 'float32')
        #print mask

        finished_seq = layers.elementwise_mul(finished_seq, mask, axis=0) + \
                        layers.elementwise_mul(alive_seq, mask2, axis = 0)
        finished_seq = layers.cast(finished_seq, 'int32')
        finished_scores = layers.elementwise_mul(finished_scores, mask, axis=0) + \
                            layers.elementwise_mul(alive_log_probs, mask2)
        finished_seq.persistable = True
        finished_scores.persistable = True

        return finished_seq, finished_scores

    finished_ids, finished_scores = beam_search(enc_output, enc_bias, source_length)
    return finished_ids, finished_scores
