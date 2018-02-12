import numpy as np

import paddle.v2 as paddle
import paddle.v2.fluid as fluid

from model import transformer, position_encoding_init
from config import TrainTaskConfig, ModelHyperParams, \
        pos_enc_param_names, input_data_names


def prepare_batch_input(insts, input_data_names, src_pad_idx, trg_pad_idx,
                        max_length, n_head, place):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias. Then, convert the numpy
    data to tensors and return a dict mapping names to tensors.
    """
    input_dict = {}

    def __pad_batch_data(insts,
                         pad_idx,
                         is_target=False,
                         return_pos=True,
                         return_attn_bias=True,
                         return_max_len=True):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []
        max_len = max(len(inst) for inst in insts)
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        if return_pos:
            inst_pos = np.array([[
                pos_i + 1 if w_i != pad_idx else 0
                for pos_i, w_i in enumerate(inst)
            ] for inst in inst_data])

            return_list += [inst_pos.astype("int64").reshape([-1, 1])]
        if return_attn_bias:
            if is_target:
                # This is used to avoid attention on paddings and subsequent
                # words.
                slf_attn_bias_data = np.ones((inst_data.shape[0], max_len,
                                              max_len))
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

    def data_to_tensor(data_list, name_list, input_dict, place):
        assert len(data_list) == len(name_list)
        for i in range(len(name_list)):
            tensor = fluid.LoDTensor()
            tensor.set(data_list[i], place)
            input_dict[name_list[i]] = tensor

    src_word, src_pos, src_slf_attn_bias, src_max_len = __pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, is_target=False)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = __pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, is_target=True)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")
    lbl_word = __pad_batch_data([inst[2] for inst in insts], trg_pad_idx, False,
                                False, False, False)

    data_to_tensor([
        src_word, src_pos, trg_word, trg_pos, src_slf_attn_bias,
        trg_slf_attn_bias, trg_src_attn_bias, lbl_word
    ], input_data_names, input_dict, place)

    return input_dict


def main():
    avg_cost = transformer(
        ModelHyperParams.src_vocab_size + 1,
        ModelHyperParams.trg_vocab_size + 1, ModelHyperParams.max_length + 1,
        ModelHyperParams.n_layer, ModelHyperParams.n_head,
        ModelHyperParams.d_key, ModelHyperParams.d_value,
        ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
        ModelHyperParams.dropout, ModelHyperParams.src_pad_idx,
        ModelHyperParams.trg_pad_idx, ModelHyperParams.pos_pad_idx)

    optimizer = fluid.optimizer.Adam(
        learning_rate=TrainTaskConfig.learning_rate,
        beta1=TrainTaskConfig.beta1,
        beta2=TrainTaskConfig.beta2,
        epsilon=TrainTaskConfig.eps)
    optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt16.train(ModelHyperParams.src_vocab_size,
                                       ModelHyperParams.trg_vocab_size),
            buf_size=51200),
        batch_size=TrainTaskConfig.batch_size)

    place = fluid.CUDAPlace(0) if TrainTaskConfig.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Initialize the parameters.
    exe.run(fluid.framework.default_startup_program())
    for pos_enc_param_name in pos_enc_param_names:
        pos_enc_param = fluid.global_scope().find_var(
            pos_enc_param_name).get_tensor()
        pos_enc_param.set(
            position_encoding_init(ModelHyperParams.max_length + 1,
                                   ModelHyperParams.d_model), place)

    for pass_id in xrange(TrainTaskConfig.pass_num):
        for batch_id, data in enumerate(train_data()):
            data_input = prepare_batch_input(
                data, input_data_names, ModelHyperParams.src_pad_idx,
                ModelHyperParams.trg_pad_idx, ModelHyperParams.max_length,
                ModelHyperParams.n_head, place)
            outs = exe.run(fluid.framework.default_main_program(),
                           feed=data_input,
                           fetch_list=[avg_cost])
            avg_cost_val = np.array(outs[0])
            print("pass_id = " + str(pass_id) + " batch = " + str(batch_id) +
                  " avg_cost = " + str(avg_cost_val))


if __name__ == "__main__":
    main()
