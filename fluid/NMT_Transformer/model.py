from functools import partial
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.layers as layers
# TODO: Remove out the batch_size from the model.
from config import batch_size, input_data_names, pos_enc_param_names


def position_encoding_init(n_position, d_pos_vec):
    """ 
    Generate the initial values for the sinusoid position encoding table.
    """
    position_enc = np.array([[
        pos / np.power(10000, 2 * (j // 2) / d_pos_vec)
        for j in range(d_pos_vec)
    ] if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
    # Set the position encoding of padding to small values rather than 0s to
    # avoid nan in attention softmax.
    position_enc[0, :] = 1e-9
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc.astype("float32")


def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         num_heads=1,
                         dropout_rate=0.):
    """
    Multi-Head Attention. Note that attn_bias will be to add to the logit to
    affect the attention weights.
    """
    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs quries, keys and values should all be 3-D tensors.")

    def __compute_qkv(queries, keys, values, num_heads, d_key, d_value):
        """ 
        Add linear projection to queries, keys, and values.
        """
        q = layers.fc(input=queries,
                      size=d_key * num_heads,
                      bias_attr=False,
                      num_flatten_dims=2)
        k = layers.fc(input=keys,
                      size=d_key * num_heads,
                      bias_attr=False,
                      num_flatten_dims=2)
        v = layers.fc(input=values,
                      size=d_value * num_heads,
                      bias_attr=False,
                      num_flatten_dims=2)
        return q, k, v

    def __split_heads(x, num_heads):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions and then transpose. Specifically, input a tensor with shape
        [bs, max_sequence_length, num_heads * hidden_dim] then output a tensor
        with shape [bs, num_heads, max_sequence_length, hidden_dim].
        """
        if num_heads == 1:
            return x

        hidden_size = x.shape[-1]
        # TODO: Decouple the program desc with batch_size.
        reshaped = layers.reshape(
            x=x, shape=[batch_size, -1, num_heads, hidden_size // num_heads])

        # permuate the dimensions into:
        # [batch_size, num_heads, max_sequence_len, hidden_size_per_head]
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
        # TODO: Decouple the program desc with batch_size.
        return layers.reshape(
            x=trans_x,
            shape=map(int,
                      [batch_size, -1, trans_x.shape[2] * trans_x.shape[3]]))

    def scaled_dot_product_attention(q, k, v, attn_bias, d_key, dropout_rate):
        """
        Scaled Dot-Product Attention
        """

        # TODO: Optimize the shape in reshape_op or softmax_op.
        # The softmax_op only supports 2D tensor currently and cann't be used
        # here. Additionally, the reshape_op cann't be used here, since the
        # shape of product inferred in compile-time is not the actual shape in
        # run-time and cann't be used to set the attribute of reshape_op. Thus,
        # define the softmax temporarily.
        def __softmax(x, eps=1e-9):
            exp_out = layers.exp(x=x)
            sum_out = layers.reduce_sum(x, dim=-1, keep_dim=False)
            return layers.elementwise_div(x=exp_out, y=sum_out, axis=0)

        scaled_q = layers.scale(x=q, scale=d_key**-0.5)
        product = layers.matmul(x=scaled_q, y=k, transpose_y=True)
        weights = __softmax(layers.elementwise_add(x=product, y=attn_bias))
        if dropout_rate:
            weights = layers.dropout(
                weights, dropout_prob=dropout_rate, is_test=False)
        out = layers.matmul(weights, v)
        return out

    q, k, v = __compute_qkv(queries, keys, values, num_heads, d_key, d_value)

    q = __split_heads(q, num_heads)
    k = __split_heads(k, num_heads)
    v = __split_heads(v, num_heads)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_key,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         bias_attr=False,
                         num_flatten_dims=2)
    return proj_out


def positionwise_feed_forward(x, d_inner_hid, d_hid):
    """
    Position-wise Feed-Forward Networks. 
    This consists of two linear transformations with a ReLU activation in
    between, which is applied to each position separately and identically.
    """
    hidden = layers.fc(input=x,
                       size=d_inner_hid,
                       bias_attr=False,
                       num_flatten_dims=2,
                       act="relu")
    out = layers.fc(input=hidden,
                    size=d_hid,
                    bias_attr=False,
                    num_flatten_dims=2)
    return out


def pre_post_process_layer(prev_out, out, process_cmd, dropout=0.):
    """
    Add residual connection, layer normalization and droput on the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    for cmd in process_cmd:
        if cmd == "a":
            out = out + prev_out if prev_out else out
        elif cmd == "n":
            out = layers.layer_norm(out, begin_norm_axis=len(out.shape) - 1)
        elif cmd == "d":
            if dropout:
                out = layers.dropout(out, dropout_prob=dropout, is_test=False)
    return out


pre_process_layer = partial(pre_post_process_layer, None)
post_process_layer = pre_post_process_layer


def prepare_encoder(src_word,
                    src_pos,
                    src_vocab_size,
                    src_emb_dim,
                    src_pad_idx,
                    src_max_len,
                    dropout=0.,
                    pos_pad_idx=0,
                    pos_enc_param_name=None):
    """
    Add word embeddings and position encodings and output a tensor with shape
    [batch_size, max_src_length_in_batch, d_model].
    This is used at the bottom of the encoder stacks.
    """
    src_word_emb = layers.embedding(
        src_word, size=[src_vocab_size, src_emb_dim], padding_idx=src_pad_idx)
    src_pos_enc = layers.embedding(
        src_pos,
        size=[src_max_len, src_emb_dim],
        param_attr=fluid.ParamAttr(
            name=pos_enc_param_name, trainable=False))
    enc_input = src_word_emb + src_pos_enc
    # TODO: Decouple the program desc with batch_size
    enc_input = layers.reshape(x=enc_input, shape=[batch_size, -1, src_emb_dim])
    return layers.dropout(
        enc_input, dropout_prob=dropout,
        is_test=False) if dropout else enc_input


prepare_encoder = partial(
    prepare_encoder, pos_enc_param_name=pos_enc_param_names[0])
prepare_decoder = partial(
    prepare_encoder, pos_enc_param_name=pos_enc_param_names[1])


def encoder_layer(enc_input, attn_bias, n_head, d_key, d_value, d_model,
                  d_inner_hid, dropout):
    """
    The layer to be stacked in the encoder.
    This consits of multi-head (self) attention followed by position-wise
    feed-forward networks and both the two components companied with the
    post_process_layer to add residual connection, layer normalization and
    droput.
    """
    attn_output = multi_head_attention(enc_input, enc_input, enc_input,
                                       attn_bias, d_key, d_value, d_model,
                                       n_head, dropout)
    attn_output = post_process_layer(enc_input, attn_output, "dan", dropout)
    ffd_output = positionwise_feed_forward(attn_output, d_inner_hid, d_model)
    output = post_process_layer(attn_output, ffd_output, "dan", dropout)
    return output


def encoder(enc_input, attn_bias, n_layer, n_head, d_key, d_value, d_model,
            d_inner_hid, dropout):
    """
    The encoder is composed of a stack of identical encoder_layer layers.
    """
    for i in range(n_layer):
        enc_output = encoder_layer(enc_input, attn_bias, n_head, d_key, d_value,
                                   d_model, d_inner_hid, dropout)
        enc_input = enc_output
    return enc_output


def decoder_layer(dec_input, enc_output, slf_attn_bias, dec_enc_attn_bias,
                  n_head, d_key, d_value, d_model, d_inner_hid, dropout):
    """
    The layer to be stacked in the decoder. The structure of this is similar to
    the encoder_layer but another multi-head attention is added to implement 
    encoder-decoder attention.
    """
    slf_attn_output = multi_head_attention(dec_input, dec_input, dec_input,
                                           slf_attn_bias, d_key, d_value,
                                           d_model, n_head, dropout)
    slf_attn_output = post_process_layer(dec_input, slf_attn_output, "dan",
                                         dropout)
    enc_attn_output = multi_head_attention(slf_attn_output, enc_output,
                                           enc_output, dec_enc_attn_bias, d_key,
                                           d_value, d_model, n_head, dropout)
    enc_attn_output = post_process_layer(slf_attn_output, enc_attn_output,
                                         "dan", dropout)
    ffd_output = positionwise_feed_forward(enc_attn_output, d_inner_hid,
                                           d_model)
    dec_output = post_process_layer(enc_attn_output, ffd_output, "dan", dropout)
    return dec_output


def decoder(dec_input, enc_output, dec_slf_attn_bias, dec_enc_attn_bias,
            n_layer, n_head, d_key, d_value, d_model, d_inner_hid, dropout):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    """
    for i in range(n_layer):
        dec_output = decoder_layer(dec_input, enc_output, dec_slf_attn_bias,
                                   dec_enc_attn_bias, n_head, d_key, d_value,
                                   d_model, d_inner_hid, dropout)
        dec_input = dec_output
    return dec_output


def transformer(src_vocab_size, trg_vocab_size, max_length, n_layer, n_head,
                d_key, d_value, d_model, d_inner_hid, dropout, src_pad_idx,
                trg_pad_idx, pos_pad_idx):
    # The shapes here only act as placeholder and are set to guarantee the
    # success of infer-shape in compile time.
    # The actual shape of src_word is:
    # [batch_size * max_src_length_in_batch, 1].
    src_word = layers.data(
        name=input_data_names[0],
        shape=[batch_size * max_length, 1],
        dtype="int64",
        append_batch_size=False)
    # The actual shape of src_pos is:
    # [batch_size * max_src_length_in_batch, 1].
    src_pos = layers.data(
        name=input_data_names[1],
        shape=[batch_size * max_length, 1],
        dtype="int64",
        append_batch_size=False)
    # The actual shape of trg_word is:
    # [batch_size * max_trg_length_in_batch, 1].
    trg_word = layers.data(
        name=input_data_names[2],
        shape=[batch_size * max_length, 1],
        dtype="int64",
        append_batch_size=False)
    # The actual shape of trg_pos is:
    # [batch_size * max_trg_length_in_batch, 1].
    trg_pos = layers.data(
        name=input_data_names[3],
        shape=[batch_size * max_length, 1],
        dtype="int64",
        append_batch_size=False)
    # The actual shape of src_slf_attn_bias is:
    # [batch_size, n_head, max_src_length_in_batch, max_src_length_in_batch].
    # This is used to avoid attention on paddings.
    src_slf_attn_bias = layers.data(
        name=input_data_names[4],
        shape=[batch_size, n_head, max_length, max_length],
        dtype="float32",
        append_batch_size=False)
    # The actual shape of trg_slf_attn_bias is:
    # [batch_size, n_head, max_trg_length_in_batch, max_trg_length_in_batch].
    # This is used to avoid attention on paddings and subsequent words.
    trg_slf_attn_bias = layers.data(
        name=input_data_names[5],
        shape=[batch_size, n_head, max_length, max_length],
        dtype="float32",
        append_batch_size=False)
    # The actual shape of trg_src_attn_bias is:
    # [batch_size, n_head, max_trg_length_in_batch, max_src_length_in_batch].
    # This is used to avoid attention on paddings.
    trg_src_attn_bias = layers.data(
        name=input_data_names[6],
        shape=[batch_size, n_head, max_length, max_length],
        dtype="float32",
        append_batch_size=False)

    enc_input = prepare_encoder(src_word, src_pos, src_vocab_size, d_model,
                                src_pad_idx, max_length, dropout)
    enc_output = encoder(enc_input, src_slf_attn_bias, n_layer, n_head, d_key,
                         d_value, d_model, d_inner_hid, dropout)

    dec_input = prepare_decoder(trg_word, trg_pos, trg_vocab_size, d_model,
                                trg_pad_idx, max_length, dropout)
    dec_output = decoder(dec_input, enc_output, trg_slf_attn_bias,
                         trg_src_attn_bias, n_layer, n_head, d_key, d_value,
                         d_model, d_inner_hid, dropout)

    # TODO: Share the same weight matrix between the two embedding layers and
    # the pre-softmax linear transformation.
    predict = layers.reshape(
        x=layers.fc(input=dec_output,
                    size=trg_vocab_size,
                    bias_attr=False,
                    num_flatten_dims=2),
        shape=[-1, trg_vocab_size],
        act="softmax")
    # The actual shape of gold is:
    # [batch_size * max_trg_length_in_batch, 1].
    gold = layers.data(
        name=input_data_names[7],
        shape=[batch_size * max_length, 1],
        dtype="int64",
        append_batch_size=False)
    cost = layers.cross_entropy(input=predict, label=gold)
    avg_cost = layers.mean(x=cost)
    return avg_cost
