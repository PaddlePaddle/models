import numpy as np

import paddle
import paddle.fluid as fluid

import model
from model import wrap_encoder as encoder
from model import wrap_decoder as decoder
from config import InferTaskConfig, ModelHyperParams, \
        encoder_input_data_names, decoder_input_data_names
from train import pad_batch_data


def translate_batch(exe,
                    src_words,
                    encoder,
                    enc_in_names,
                    enc_out_names,
                    decoder,
                    dec_in_names,
                    dec_out_names,
                    beam_size,
                    max_length,
                    n_best,
                    batch_size,
                    n_head,
                    d_model,
                    src_pad_idx,
                    trg_pad_idx,
                    bos_idx,
                    eos_idx,
                    unk_idx,
                    output_unk=True):
    """
    Run the encoder program once and run the decoder program multiple times to
    implement beam search externally.
    """
    # Prepare data for encoder and run the encoder.
    enc_in_data = pad_batch_data(
        src_words,
        src_pad_idx,
        n_head,
        is_target=False,
        is_label=False,
        return_attn_bias=True,
        return_max_len=False)
    # Append the data shape input to reshape the output of embedding layer.
    enc_in_data = enc_in_data + [
        np.array(
            [-1, enc_in_data[2].shape[-1], d_model], dtype="int32")
    ]
    # Append the shape inputs to reshape before and after softmax in encoder
    # self attention.
    enc_in_data = enc_in_data + [
        np.array(
            [-1, enc_in_data[2].shape[-1]], dtype="int32"), np.array(
                enc_in_data[2].shape, dtype="int32")
    ]
    enc_output = exe.run(encoder,
                         feed=dict(zip(enc_in_names, enc_in_data)),
                         fetch_list=enc_out_names)[0]

    # Beam Search.
    # To store the beam info.
    scores = np.zeros((batch_size, beam_size), dtype="float32")
    prev_branchs = [[] for i in range(batch_size)]
    next_ids = [[] for i in range(batch_size)]
    # Use beam_inst_map to map beam idx to the instance idx in batch, since the
    # size of feeded batch is changing.
    beam_inst_map = {
        beam_idx: inst_idx
        for inst_idx, beam_idx in enumerate(range(batch_size))
    }
    # Use active_beams to recode the alive.
    active_beams = range(batch_size)

    def beam_backtrace(prev_branchs, next_ids, n_best=beam_size):
        """
        Decode and select n_best sequences for one instance by backtrace.
        """
        seqs = []
        for i in range(n_best):
            k = i
            seq = []
            for j in range(len(prev_branchs) - 1, -1, -1):
                seq.append(next_ids[j][k])
                k = prev_branchs[j][k]
            seq = seq[::-1]
            # Add the <bos>, since next_ids don't include the <bos>.
            seq = [bos_idx] + seq
            seqs.append(seq)
        return seqs

    def init_dec_in_data(batch_size, beam_size, enc_in_data, enc_output):
        """
        Initialize the input data for decoder.
        """
        trg_words = np.array(
            [[bos_idx]] * batch_size * beam_size, dtype="int64")
        trg_pos = np.array([[1]] * batch_size * beam_size, dtype="int64")
        src_max_length, src_slf_attn_bias, trg_max_len = enc_in_data[2].shape[
            -1], enc_in_data[2], 1
        # This is used to remove attention on subsequent words.
        trg_slf_attn_bias = np.ones((batch_size * beam_size, trg_max_len,
                                     trg_max_len))
        trg_slf_attn_bias = np.triu(trg_slf_attn_bias, 1).reshape(
            [-1, 1, trg_max_len, trg_max_len])
        trg_slf_attn_bias = (np.tile(trg_slf_attn_bias, [1, n_head, 1, 1]) *
                             [-1e9]).astype("float32")
        # This is used to remove attention on the paddings of source sequences.
        trg_src_attn_bias = np.tile(
            src_slf_attn_bias[:, :, ::src_max_length, :][:, np.newaxis],
            [1, beam_size, 1, trg_max_len, 1]).reshape([
                -1, src_slf_attn_bias.shape[1], trg_max_len,
                src_slf_attn_bias.shape[-1]
            ])
        # Append the shape input to reshape the output of embedding layer.
        trg_data_shape = np.array(
            [batch_size * beam_size, trg_max_len, d_model], dtype="int32")
        # Append the shape inputs to reshape before and after softmax in
        # decoder self attention.
        trg_slf_attn_pre_softmax_shape = np.array(
            [-1, trg_slf_attn_bias.shape[-1]], dtype="int32")
        trg_slf_attn_post_softmax_shape = np.array(
            trg_slf_attn_bias.shape, dtype="int32")
        # Append the shape inputs to reshape before and after softmax in
        # encoder-decoder attention.
        trg_src_attn_pre_softmax_shape = np.array(
            [-1, trg_src_attn_bias.shape[-1]], dtype="int32")
        trg_src_attn_post_softmax_shape = np.array(
            trg_src_attn_bias.shape, dtype="int32")
        enc_output = np.tile(
            enc_output[:, np.newaxis], [1, beam_size, 1, 1]).reshape(
                [-1, enc_output.shape[-2], enc_output.shape[-1]])
        return trg_words, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, \
            trg_data_shape, trg_slf_attn_pre_softmax_shape, \
            trg_slf_attn_post_softmax_shape, trg_src_attn_pre_softmax_shape, \
            trg_src_attn_post_softmax_shape, enc_output

    def update_dec_in_data(dec_in_data, next_ids, active_beams, beam_inst_map):
        """
        Update the input data of decoder mainly by slicing from the previous
        input data and dropping the finished instance beams.
        """
        trg_words, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, \
            trg_data_shape, trg_slf_attn_pre_softmax_shape, \
            trg_slf_attn_post_softmax_shape, trg_src_attn_pre_softmax_shape, \
            trg_src_attn_post_softmax_shape, enc_output = dec_in_data
        trg_cur_len = trg_slf_attn_bias.shape[-1] + 1
        trg_words = np.array(
            [
                beam_backtrace(prev_branchs[beam_idx], next_ids[beam_idx])
                for beam_idx in active_beams
            ],
            dtype="int64")
        trg_words = trg_words.reshape([-1, 1])
        trg_pos = np.array(
            [range(1, trg_cur_len + 1)] * len(active_beams) * beam_size,
            dtype="int64").reshape([-1, 1])
        active_beams = [beam_inst_map[beam_idx] for beam_idx in active_beams]
        active_beams_indice = (
            (np.array(active_beams) * beam_size)[:, np.newaxis] +
            np.array(range(beam_size))[np.newaxis, :]).flatten()
        # This is used to remove attention on subsequent words.
        trg_slf_attn_bias = np.ones((len(active_beams) * beam_size, trg_cur_len,
                                     trg_cur_len))
        trg_slf_attn_bias = np.triu(trg_slf_attn_bias, 1).reshape(
            [-1, 1, trg_cur_len, trg_cur_len])
        trg_slf_attn_bias = (np.tile(trg_slf_attn_bias, [1, n_head, 1, 1]) *
                             [-1e9]).astype("float32")
        # This is used to remove attention on the paddings of source sequences.
        trg_src_attn_bias = np.tile(trg_src_attn_bias[
            active_beams_indice, :, ::trg_src_attn_bias.shape[2], :],
                                    [1, 1, trg_cur_len, 1])
        # Append the shape input to reshape the output of embedding layer.
        trg_data_shape = np.array(
            [len(active_beams) * beam_size, trg_cur_len, d_model],
            dtype="int32")
        # Append the shape inputs to reshape before and after softmax in
        # decoder self attention.
        trg_slf_attn_pre_softmax_shape = np.array(
            [-1, trg_slf_attn_bias.shape[-1]], dtype="int32")
        trg_slf_attn_post_softmax_shape = np.array(
            trg_slf_attn_bias.shape, dtype="int32")
        # Append the shape inputs to reshape before and after softmax in
        # encoder-decoder attention.
        trg_src_attn_pre_softmax_shape = np.array(
            [-1, trg_src_attn_bias.shape[-1]], dtype="int32")
        trg_src_attn_post_softmax_shape = np.array(
            trg_src_attn_bias.shape, dtype="int32")
        enc_output = enc_output[active_beams_indice, :, :]
        return trg_words, trg_pos, trg_slf_attn_bias, trg_src_attn_bias, \
            trg_data_shape, trg_slf_attn_pre_softmax_shape, \
            trg_slf_attn_post_softmax_shape, trg_src_attn_pre_softmax_shape, \
            trg_src_attn_post_softmax_shape, enc_output

    dec_in_data = init_dec_in_data(batch_size, beam_size, enc_in_data,
                                   enc_output)
    for i in range(max_length):
        predict_all = exe.run(decoder,
                              feed=dict(zip(dec_in_names, dec_in_data)),
                              fetch_list=dec_out_names)[0]
        predict_all = np.log(
            predict_all.reshape([len(beam_inst_map) * beam_size, i + 1, -1])
            [:, -1, :])
        predict_all = (predict_all + scores[active_beams].reshape(
            [len(beam_inst_map) * beam_size, -1])).reshape(
                [len(beam_inst_map), beam_size, -1])
        if not output_unk:  # To exclude the <unk> token.
            predict_all[:, :, unk_idx] = -1e9
        active_beams = []
        for beam_idx in range(batch_size):
            if not beam_inst_map.has_key(beam_idx):
                continue
            inst_idx = beam_inst_map[beam_idx]
            predict = (predict_all[inst_idx, :, :]
                       if i != 0 else predict_all[inst_idx, 0, :]).flatten()
            top_k_indice = np.argpartition(predict, -beam_size)[-beam_size:]
            top_scores_ids = top_k_indice[np.argsort(predict[top_k_indice])[::
                                                                            -1]]
            top_scores = predict[top_scores_ids]
            scores[beam_idx] = top_scores
            prev_branchs[beam_idx].append(top_scores_ids /
                                          predict_all.shape[-1])
            next_ids[beam_idx].append(top_scores_ids % predict_all.shape[-1])
            if next_ids[beam_idx][-1][0] != eos_idx:
                active_beams.append(beam_idx)
        if len(active_beams) == 0:
            break
        dec_in_data = update_dec_in_data(dec_in_data, next_ids, active_beams,
                                         beam_inst_map)
        beam_inst_map = {
            beam_idx: inst_idx
            for inst_idx, beam_idx in enumerate(active_beams)
        }

    # Decode beams and select n_best sequences for each instance by backtrace.
    seqs = [
        beam_backtrace(prev_branchs[beam_idx], next_ids[beam_idx], n_best)
        for beam_idx in range(batch_size)
    ]

    return seqs, scores[:, :n_best].tolist()


def main():
    place = fluid.CUDAPlace(0) if InferTaskConfig.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    encoder_program = fluid.Program()
    with fluid.program_guard(main_program=encoder_program):
        enc_output = encoder(
            ModelHyperParams.src_vocab_size, ModelHyperParams.max_length + 1,
            ModelHyperParams.n_layer, ModelHyperParams.n_head,
            ModelHyperParams.d_key, ModelHyperParams.d_value,
            ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
            ModelHyperParams.dropout)

    decoder_program = fluid.Program()
    with fluid.program_guard(main_program=decoder_program):
        predict = decoder(
            ModelHyperParams.trg_vocab_size, ModelHyperParams.max_length + 1,
            ModelHyperParams.n_layer, ModelHyperParams.n_head,
            ModelHyperParams.d_key, ModelHyperParams.d_value,
            ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
            ModelHyperParams.dropout)

    # Load model parameters of encoder and decoder separately from the saved
    # transformer model.
    encoder_var_names = []
    for op in encoder_program.block(0).ops:
        encoder_var_names += op.input_arg_names
    encoder_param_names = filter(
        lambda var_name: isinstance(encoder_program.block(0).var(var_name),
            fluid.framework.Parameter),
        encoder_var_names)
    encoder_params = map(encoder_program.block(0).var, encoder_param_names)
    decoder_var_names = []
    for op in decoder_program.block(0).ops:
        decoder_var_names += op.input_arg_names
    decoder_param_names = filter(
        lambda var_name: isinstance(decoder_program.block(0).var(var_name),
            fluid.framework.Parameter),
        decoder_var_names)
    decoder_params = map(decoder_program.block(0).var, decoder_param_names)
    fluid.io.load_vars(exe, InferTaskConfig.model_path, vars=encoder_params)
    fluid.io.load_vars(exe, InferTaskConfig.model_path, vars=decoder_params)

    # This is used here to set dropout to the test mode.
    encoder_program = fluid.io.get_inference_program(
        target_vars=[enc_output], main_program=encoder_program)
    decoder_program = fluid.io.get_inference_program(
        target_vars=[predict], main_program=decoder_program)

    test_data = paddle.batch(
        paddle.dataset.wmt16.test(ModelHyperParams.src_vocab_size,
                                  ModelHyperParams.trg_vocab_size),
        batch_size=InferTaskConfig.batch_size)

    trg_idx2word = paddle.dataset.wmt16.get_dict(
        "de", dict_size=ModelHyperParams.trg_vocab_size, reverse=True)

    def post_process_seq(seq,
                         bos_idx=ModelHyperParams.bos_idx,
                         eos_idx=ModelHyperParams.eos_idx,
                         output_bos=InferTaskConfig.output_bos,
                         output_eos=InferTaskConfig.output_eos):
        """
        Post-process the beam-search decoded sequence. Truncate from the first
        <eos> and remove the <bos> and <eos> tokens currently.
        """
        eos_pos = len(seq) - 1
        for i, idx in enumerate(seq):
            if idx == eos_idx:
                eos_pos = i
                break
        seq = seq[:eos_pos + 1]
        return filter(
            lambda idx: (output_bos or idx != bos_idx) and \
                (output_eos or idx != eos_idx),
            seq)

    for batch_id, data in enumerate(test_data()):
        batch_seqs, batch_scores = translate_batch(
            exe,
            [item[0] for item in data],
            encoder_program,
            encoder_input_data_names,
            [enc_output.name],
            decoder_program,
            decoder_input_data_names,
            [predict.name],
            InferTaskConfig.beam_size,
            InferTaskConfig.max_length,
            InferTaskConfig.n_best,
            len(data),
            ModelHyperParams.n_head,
            ModelHyperParams.d_model,
            ModelHyperParams.eos_idx,  # Use eos_idx to pad.
            ModelHyperParams.eos_idx,  # Use eos_idx to pad.
            ModelHyperParams.bos_idx,
            ModelHyperParams.eos_idx,
            ModelHyperParams.unk_idx,
            output_unk=InferTaskConfig.output_unk)
        for i in range(len(batch_seqs)):
            # Post-process the beam-search decoded sequences.
            seqs = map(post_process_seq, batch_seqs[i])
            scores = batch_scores[i]
            for seq in seqs:
                print(" ".join([trg_idx2word[idx] for idx in seq]))


if __name__ == "__main__":
    main()
