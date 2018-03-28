import paddle.v2 as paddle
from paddle.v2.layer import parse_network

__all__ = ["encoder_decoder_network"]


def _bidirect_lstm_encoder(input, hidden_dim, depth):
    lstm_last = []
    for dirt in ["fwd", "bwd"]:
        for i in range(depth):
            input_proj = paddle.layer.mixed(
                name="__in_proj_%0d_%s__" % (i, dirt),
                size=hidden_dim * 4,
                bias_attr=True,
                input=[
                    paddle.layer.full_matrix_projection(input_proj),
                    paddle.layer.full_matrix_projection(
                        lstm, param_attr=paddle.attr.Param(initial_std=5e-4)),
                ] if i else [paddle.layer.full_matrix_projection(input)])
            lstm = paddle.layer.lstmemory(
                input=input_proj,
                bias_attr=paddle.attr.Param(initial_std=0.),
                param_attr=paddle.attr.Param(initial_std=5e-4),
                reverse=i % 2 if dirt == "fwd" else not i % 2)
        lstm_last.append(lstm)
    return paddle.layer.concat(input=lstm_last)


def _attended_decoder_step(word_count, enc_out, enc_out_proj,
                           decoder_hidden_dim, depth, trg_emb):
    decoder_memory = paddle.layer.memory(
        name="__decoder_0__", size=decoder_hidden_dim, boot_layer=None)

    context = paddle.networks.simple_attention(
        encoded_sequence=enc_out,
        encoded_proj=enc_out_proj,
        decoder_state=decoder_memory)

    for i in range(depth):
        input_proj = paddle.layer.mixed(
            act=paddle.activation.Linear(),
            size=decoder_hidden_dim * 4,
            bias_attr=False,
            input=[
                paddle.layer.full_matrix_projection(input_proj),
                paddle.layer.full_matrix_projection(lstm)
            ] if i else [
                paddle.layer.full_matrix_projection(context),
                paddle.layer.full_matrix_projection(trg_emb)
            ])
        lstm = paddle.networks.lstmemory_unit(
            input=input_proj,
            input_proj_layer_attr=paddle.attr.ExtraLayerAttribute(
                error_clipping_threshold=25.),
            out_memory=decoder_memory if not i else None,
            name="__decoder_%d__" % (i),
            size=decoder_hidden_dim,
            act=paddle.activation.Tanh(),
            gate_act=paddle.activation.Sigmoid(),
            state_act=paddle.activation.Tanh())

    next_word = paddle.layer.fc(size=word_count,
                                bias_attr=True,
                                act=paddle.activation.Softmax(),
                                input=lstm)
    return next_word


def encoder_decoder_network(word_count,
                            emb_dim,
                            encoder_depth,
                            encoder_hidden_dim,
                            decoder_depth,
                            decoder_hidden_dim,
                            bos_id,
                            eos_id,
                            max_length,
                            beam_size=10,
                            is_generating=False):
    src_emb = paddle.layer.embedding(
        input=paddle.layer.data(
            name="src_word_id",
            type=paddle.data_type.integer_value_sequence(word_count)),
        size=emb_dim,
        param_attr=paddle.attr.ParamAttr(name="__embedding__"))
    enc_out = _bidirect_lstm_encoder(
        input=src_emb, hidden_dim=encoder_hidden_dim, depth=encoder_depth)
    enc_out_proj = paddle.layer.fc(act=paddle.activation.Linear(),
                                   size=encoder_hidden_dim,
                                   bias_attr=False,
                                   input=enc_out)

    decoder_group_name = "decoder_group"
    group_inputs = [
        word_count, paddle.layer.StaticInput(input=enc_out),
        paddle.layer.StaticInput(input=enc_out_proj), decoder_hidden_dim,
        decoder_depth
    ]

    if is_generating:
        gen_trg_emb = paddle.layer.GeneratedInput(
            size=word_count,
            embedding_name="__embedding__",
            embedding_size=emb_dim)
        return paddle.layer.beam_search(
            name=decoder_group_name,
            step=_attended_decoder_step,
            input=group_inputs + [gen_trg_emb],
            bos_id=bos_id,
            eos_id=eos_id,
            beam_size=beam_size,
            max_length=max_length)

    else:
        trg_emb = paddle.layer.embedding(
            input=paddle.layer.data(
                name="trg_word_id",
                type=paddle.data_type.integer_value_sequence(word_count)),
            size=emb_dim,
            param_attr=paddle.attr.ParamAttr(name="__embedding__"))
        lbl = paddle.layer.data(
            name="trg_next_word",
            type=paddle.data_type.integer_value_sequence(word_count))
        next_word = paddle.layer.recurrent_group(
            name=decoder_group_name,
            step=_attended_decoder_step,
            input=group_inputs + [trg_emb])
        return paddle.layer.classification_cost(input=next_word, label=lbl)
