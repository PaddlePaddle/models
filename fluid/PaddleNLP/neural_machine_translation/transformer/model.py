from functools import partial
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers

from config import *


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(np.arange(
        num_timescales)) * -log_timescale_increment
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
                         cache=None):
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
        scaled_q = layers.scale(x=q, scale=d_key**-0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        if attn_bias:
            product += attn_bias
        weights = layers.softmax(product)
        if dropout_rate:
            weights = layers.dropout(
                weights,
                dropout_prob=dropout_rate,
                seed=ModelHyperParams.dropout_seed,
                is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    if cache is not None:  # use cache and concat time steps
        # Since the inplace reshape in __split_heads changes the shape of k and
        # v, which is the cache input for next time step, reshape the cache
        # input from the previous time step first.
        k = cache["k"] = layers.concat(
            [layers.reshape(
                cache["k"], shape=[0, 0, d_key * n_head]), k],
            axis=1)
        v = cache["v"] = layers.concat(
            [layers.reshape(
                cache["v"], shape=[0, 0, d_value * n_head]), v],
            axis=1)

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_model,
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
            seed=ModelHyperParams.dropout_seed,
            is_test=False)
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
            out = layers.layer_norm(
                out,
                begin_norm_axis=len(out.shape) - 1,
                param_attr=fluid.initializer.Constant(1.),
                bias_attr=fluid.initializer.Constant(0.))
        elif cmd == "d":  # add dropout
            if dropout_rate:
                out = layers.dropout(
                    out,
                    dropout_prob=dropout_rate,
                    seed=ModelHyperParams.dropout_seed,
                    is_test=False)
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
                            pos_enc_param_name=None):
    """Add word embeddings and position encodings.
    The output tensor has a shape of:
    [batch_size, max_src_length_in_batch, d_model].
    This module is used at the bottom of the encoder stacks.
    """
    src_word_emb = layers.embedding(
        src_word,
        size=[src_vocab_size, src_emb_dim],
        padding_idx=ModelHyperParams.bos_idx,  # set embedding of bos to 0
        param_attr=fluid.ParamAttr(
            name=word_emb_param_name,
            initializer=fluid.initializer.Normal(0., src_emb_dim**-0.5)))

    src_word_emb = layers.scale(x=src_word_emb, scale=src_emb_dim**0.5)
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            name=pos_enc_param_name, trainable=False))
    src_pos_enc.stop_gradient = True
    enc_input = src_word_emb + src_pos_enc
    return layers.dropout(
        enc_input,
        dropout_prob=dropout_rate,
        seed=ModelHyperParams.dropout_seed,
        is_test=False) if dropout_rate else enc_input


prepare_encoder = partial(
    prepare_encoder_decoder, pos_enc_param_name=pos_enc_param_names[0])
prepare_decoder = partial(
    prepare_encoder_decoder, pos_enc_param_name=pos_enc_param_names[1])


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
        enc_input = enc_output
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
    reader = layers.py_reader(
        capacity=20,
        name="test_reader" if is_test else "train_reader",
        shapes=[input_descs[input_field][0] for input_field in input_fields],
        dtypes=[input_descs[input_field][1] for input_field in input_fields],
        lod_levels=[
            input_descs[input_field][2]
            if len(input_descs[input_field]) == 3 else 0
            for input_field in input_fields
        ])
    return layers.read_file(reader), reader


def transformer(src_vocab_size,
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
                label_smooth_eps,
                use_py_reader=False,
                is_test=False):
    if weight_sharing:
        assert src_vocab_size == trg_vocab_size, (
            "Vocabularies in source and target should be same for weight sharing."
        )

    data_input_names = encoder_data_input_fields + \
                decoder_data_input_fields[:-1] + label_data_input_fields

    if use_py_reader:
        all_inputs, reader = make_all_py_reader_inputs(data_input_names,
                                                       is_test)
    else:
        all_inputs = make_all_inputs(data_input_names)

    enc_inputs_len = len(encoder_data_input_fields)
    dec_inputs_len = len(decoder_data_input_fields[:-1])
    enc_inputs = all_inputs[0:enc_inputs_len]
    dec_inputs = all_inputs[enc_inputs_len:enc_inputs_len + dec_inputs_len]
    label = all_inputs[-2]
    weights = all_inputs[-1]

    enc_output = wrap_encoder(
        src_vocab_size,
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
        enc_inputs, )

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
        dec_inputs,
        enc_output, )

    # Padding index do not contribute to the total loss. The weights is used to
    # cancel padding index in calculating the loss.
    if label_smooth_eps:
        label = layers.label_smooth(
            label=layers.one_hot(
                input=label, depth=trg_vocab_size),
            epsilon=label_smooth_eps)

    cost = layers.softmax_with_cross_entropy(
        logits=predict,
        label=label,
        soft_label=True if label_smooth_eps else False)
    weighted_cost = cost * weights
    sum_cost = layers.reduce_sum(weighted_cost)
    token_num = layers.reduce_sum(weights)
    token_num.stop_gradient = True
    avg_cost = sum_cost / token_num
    return sum_cost, avg_cost, predict, token_num, reader if use_py_reader else None


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
                 enc_inputs=None):
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
        word_emb_param_name=word_emb_param_names[0])
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
                 dec_inputs=None,
                 enc_output=None,
                 caches=None):
    """
    The wrapper assembles together all needed layers for the decoder.
    """
    if dec_inputs is None:
        # This is used to implement independent decoder program in inference.
        trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, enc_output = \
            make_all_inputs(decoder_data_input_fields)
    else:
        trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias = dec_inputs

    dec_input = prepare_decoder(
        trg_word,
        trg_pos,
        trg_vocab_size,
        d_model,
        max_length,
        prepostprocess_dropout,
        word_emb_param_name=word_emb_param_names[0]
        if weight_sharing else word_emb_param_names[1])
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
    if weight_sharing:
        predict = layers.matmul(
            x=dec_output,
            y=fluid.default_main_program().global_block().var(
                word_emb_param_names[0]),
            transpose_y=True)
    else:
        predict = layers.fc(input=dec_output,
                            size=trg_vocab_size,
                            bias_attr=False)
    if dec_inputs is None:
        # Return probs for independent decoder program.
        predict = layers.softmax(predict)
    return predict


def fast_decode(
        src_vocab_size,
        trg_vocab_size,
        max_in_len,
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
        beam_size,
        max_out_len,
        eos_idx, ):
    """
    Use beam search to decode. Caches will be used to store states of history
    steps which can make the decoding faster.
    """
    enc_output = wrap_encoder(
        src_vocab_size, max_in_len, n_layer, n_head, d_key, d_value, d_model,
        d_inner_hid, prepostprocess_dropout, attention_dropout, relu_dropout,
        preprocess_cmd, postprocess_cmd, weight_sharing)
    start_tokens, init_scores, trg_src_attn_bias = make_all_inputs(
        fast_decoder_data_input_fields)

    def beam_search():
        max_len = layers.fill_constant(
            shape=[1], dtype=start_tokens.dtype, value=max_out_len)
        step_idx = layers.fill_constant(
            shape=[1], dtype=start_tokens.dtype, value=0)
        cond = layers.less_than(x=step_idx, y=max_len)
        while_op = layers.While(cond)
        # array states will be stored for each step.
        ids = layers.array_write(
            layers.reshape(start_tokens, (-1, 1)), step_idx)
        scores = layers.array_write(init_scores, step_idx)
        # cell states will be overwrited at each step.
        # caches contains states of history steps to reduce redundant
        # computation in decoder.
        caches = [{
            "k": layers.fill_constant_batch_size_like(
                input=start_tokens,
                shape=[-1, 0, d_model],
                dtype=enc_output.dtype,
                value=0),
            "v": layers.fill_constant_batch_size_like(
                input=start_tokens,
                shape=[-1, 0, d_model],
                dtype=enc_output.dtype,
                value=0)
        } for i in range(n_layer)]
        with while_op.block():
            pre_ids = layers.array_read(array=ids, i=step_idx)
            pre_ids = layers.reshape(pre_ids, (-1, 1, 1))
            pre_scores = layers.array_read(array=scores, i=step_idx)
            # sequence_expand can gather sequences according to lod thus can be
            # used in beam search to sift states corresponding to selected ids.
            pre_src_attn_bias = layers.sequence_expand(
                x=trg_src_attn_bias, y=pre_scores)
            pre_enc_output = layers.sequence_expand(x=enc_output, y=pre_scores)
            pre_caches = [{
                "k": layers.sequence_expand(
                    x=cache["k"], y=pre_scores),
                "v": layers.sequence_expand(
                    x=cache["v"], y=pre_scores),
            } for cache in caches]
            pre_pos = layers.elementwise_mul(
                x=layers.fill_constant_batch_size_like(
                    input=pre_enc_output,  # cann't use pre_ids here since it has lod
                    value=1,
                    shape=[-1, 1, 1],
                    dtype=pre_ids.dtype),
                y=step_idx,
                axis=0)
            logits = wrap_decoder(
                trg_vocab_size,
                max_in_len,
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
                dec_inputs=(pre_ids, pre_pos, None, pre_src_attn_bias),
                enc_output=pre_enc_output,
                caches=pre_caches)

            topk_scores, topk_indices = layers.topk(
                input=layers.softmax(logits), k=beam_size)
            accu_scores = layers.elementwise_add(
                x=layers.log(topk_scores),
                y=layers.reshape(
                    pre_scores, shape=[-1]),
                axis=0)
            # beam_search op uses lod to distinguish branches.
            topk_indices = layers.lod_reset(topk_indices, pre_ids)
            selected_ids, selected_scores = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=beam_size,
                end_id=eos_idx)

            layers.increment(x=step_idx, value=1.0, in_place=True)
            # update states
            layers.array_write(selected_ids, i=step_idx, array=ids)
            layers.array_write(selected_scores, i=step_idx, array=scores)
            layers.assign(pre_src_attn_bias, trg_src_attn_bias)
            layers.assign(pre_enc_output, enc_output)
            for i in range(n_layer):
                layers.assign(pre_caches[i]["k"], caches[i]["k"])
                layers.assign(pre_caches[i]["v"], caches[i]["v"])
            length_cond = layers.less_than(x=step_idx, y=max_len)
            finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
            layers.logical_and(x=length_cond, y=finish_cond, out=cond)

        finished_ids, finished_scores = layers.beam_search_decode(
            ids, scores, beam_size=beam_size, end_id=eos_idx)
        return finished_ids, finished_scores

    finished_ids, finished_scores = beam_search()
    return finished_ids, finished_scores
