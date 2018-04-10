import os
import time
import numpy as np

import paddle
import paddle.fluid as fluid

from model import transformer, position_encoding_init
from optim import LearningRateScheduler
from config import TrainTaskConfig, ModelHyperParams, pos_enc_param_names, \
        encoder_input_data_names, decoder_input_data_names, label_data_names


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array(
            [[1.] * len(inst) + [0.] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            range(1, len(inst) + 1) + [0] * (max_len - len(inst))
            for inst in insts
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones((inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data, 1).reshape(
                [-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_input(insts, input_data_names, src_pad_idx, trg_pad_idx,
                        n_head, d_model):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    # These shape tensors are used in reshape_op.
    src_data_shape = np.array([len(insts), src_max_len, d_model], dtype="int32")
    trg_data_shape = np.array([len(insts), trg_max_len, d_model], dtype="int32")
    src_slf_attn_pre_softmax_shape = np.array(
        [-1, src_slf_attn_bias.shape[-1]], dtype="int32")
    src_slf_attn_post_softmax_shape = np.array(
        src_slf_attn_bias.shape, dtype="int32")
    trg_slf_attn_pre_softmax_shape = np.array(
        [-1, trg_slf_attn_bias.shape[-1]], dtype="int32")
    trg_slf_attn_post_softmax_shape = np.array(
        trg_slf_attn_bias.shape, dtype="int32")
    trg_src_attn_pre_softmax_shape = np.array(
        [-1, trg_src_attn_bias.shape[-1]], dtype="int32")
    trg_src_attn_post_softmax_shape = np.array(
        trg_src_attn_bias.shape, dtype="int32")

    lbl_word, lbl_weight = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False)

    input_dict = dict(
        zip(input_data_names, [
            src_word, src_pos, src_slf_attn_bias, src_data_shape,
            src_slf_attn_pre_softmax_shape, src_slf_attn_post_softmax_shape,
            trg_word, trg_pos, trg_slf_attn_bias, trg_src_attn_bias,
            trg_data_shape, trg_slf_attn_pre_softmax_shape,
            trg_slf_attn_post_softmax_shape, trg_src_attn_pre_softmax_shape,
            trg_src_attn_post_softmax_shape, lbl_word, lbl_weight
        ]))
    return input_dict


def main():
    place = fluid.CUDAPlace(0) if TrainTaskConfig.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    sum_cost, avg_cost, predict, token_num = transformer(
        ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
        ModelHyperParams.n_head, ModelHyperParams.d_key,
        ModelHyperParams.d_value, ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid, ModelHyperParams.dropout)

    lr_scheduler = LearningRateScheduler(ModelHyperParams.d_model,
                                         TrainTaskConfig.warmup_steps, place,
                                         TrainTaskConfig.learning_rate)
    optimizer = fluid.optimizer.Adam(
        learning_rate=lr_scheduler.learning_rate,
        beta1=TrainTaskConfig.beta1,
        beta2=TrainTaskConfig.beta2,
        epsilon=TrainTaskConfig.eps)
    optimizer.minimize(avg_cost if TrainTaskConfig.use_avg_cost else sum_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt16.train(ModelHyperParams.src_vocab_size,
                                       ModelHyperParams.trg_vocab_size),
            buf_size=100000),
        batch_size=TrainTaskConfig.batch_size)

    # Program to do validation.
    test_program = fluid.default_main_program().clone()
    with fluid.program_guard(test_program):
        test_program = fluid.io.get_inference_program([avg_cost])
    val_data = paddle.batch(
        paddle.dataset.wmt16.validation(ModelHyperParams.src_vocab_size,
                                        ModelHyperParams.trg_vocab_size),
        batch_size=TrainTaskConfig.batch_size)

    def test(exe):
        test_total_cost = 0
        test_total_token = 0
        for batch_id, data in enumerate(val_data()):
            data_input = prepare_batch_input(
                data, encoder_input_data_names + decoder_input_data_names[:-1] +
                label_data_names, ModelHyperParams.eos_idx,
                ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            test_sum_cost, test_token_num = exe.run(
                test_program,
                feed=data_input,
                fetch_list=[sum_cost, token_num],
                use_program_cache=True)
            test_total_cost += test_sum_cost
            test_total_token += test_token_num
        test_avg_cost = test_total_cost / test_total_token
        test_ppl = np.exp([min(test_avg_cost, 100)])
        return test_avg_cost, test_ppl

    # Initialize the parameters.
    exe.run(fluid.framework.default_startup_program())
    for pos_enc_param_name in pos_enc_param_names:
        pos_enc_param = fluid.global_scope().find_var(
            pos_enc_param_name).get_tensor()
        pos_enc_param.set(
            position_encoding_init(ModelHyperParams.max_length + 1,
                                   ModelHyperParams.d_model), place)

    for pass_id in xrange(TrainTaskConfig.pass_num):
        pass_start_time = time.time()
        for batch_id, data in enumerate(train_data()):
            if len(data) != TrainTaskConfig.batch_size:
                continue
            data_input = prepare_batch_input(
                data, encoder_input_data_names + decoder_input_data_names[:-1] +
                label_data_names, ModelHyperParams.eos_idx,
                ModelHyperParams.eos_idx, ModelHyperParams.n_head,
                ModelHyperParams.d_model)
            lr_scheduler.update_learning_rate(data_input)
            outs = exe.run(fluid.framework.default_main_program(),
                           feed=data_input,
                           fetch_list=[sum_cost, avg_cost],
                           use_program_cache=True)
            sum_cost_val, avg_cost_val = np.array(outs[0]), np.array(outs[1])
            print("epoch: %d, batch: %d, sum loss: %f, avg loss: %f, ppl: %f" %
                  (pass_id, batch_id, sum_cost_val, avg_cost_val,
                   np.exp([min(avg_cost_val[0], 100)])))
        # Validate and save the model for inference.
        val_avg_cost, val_ppl = test(exe)
        pass_end_time = time.time()
        time_consumed = pass_end_time - pass_start_time
        print("epoch: %d, val avg loss: %f, val ppl: %f, "
              "consumed %fs" % (pass_id, val_avg_cost, val_ppl, time_consumed))
        fluid.io.save_inference_model(
            os.path.join(TrainTaskConfig.model_dir,
                         "pass_" + str(pass_id) + ".infer.model"),
            encoder_input_data_names + decoder_input_data_names[:-1],
            [predict], exe)


if __name__ == "__main__":
    main()
