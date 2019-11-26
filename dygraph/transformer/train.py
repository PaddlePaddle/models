# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import argparse
import ast

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.dataset.wmt16 as wmt16

from model import TransFormer, NoamDecay
from config import *
from data_util import *


def parse_args():
    parser = argparse.ArgumentParser("Arguments for Training")
    parser.add_argument(
        "--use_data_parallel",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use multi-GPU.")
    parser.add_argument(
        "--model_file",
        type=str,
        default="transformer_params",
        help="Save the model as a file named `model_file.pdparams`.")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    merge_cfg_from_list(args.opts, [TrainTaskConfig, ModelHyperParams])
    return args


def prepare_train_input(insts, src_pad_idx, trg_pad_idx, n_head):
    """
    inputs for training
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len, 1)
    trg_pos = trg_pos.reshape(-1, trg_max_len, 1)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[2] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)

    data_inputs = [
        src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
        trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
    ]

    var_inputs = []
    for i, field in enumerate(encoder_data_input_fields +
                              decoder_data_input_fields[:-1] +
                              label_data_input_fields):
        var_inputs.append(to_variable(data_inputs[i], name=field))

    enc_inputs = var_inputs[0:len(encoder_data_input_fields)]
    dec_inputs = var_inputs[len(encoder_data_input_fields
                                ):len(encoder_data_input_fields) +
                            len(decoder_data_input_fields[:-1])]
    label = var_inputs[-2]
    weights = var_inputs[-1]

    return enc_inputs, dec_inputs, label, weights


def train(args):
    """
    train models
    :return:
    """

    trainer_count = fluid.dygraph.parallel.Env().nranks
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if args.use_data_parallel else fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        # define model
        transformer = TransFormer(
            'transformer', ModelHyperParams.src_vocab_size,
            ModelHyperParams.trg_vocab_size, ModelHyperParams.max_length + 1,
            ModelHyperParams.n_layer, ModelHyperParams.n_head,
            ModelHyperParams.d_key, ModelHyperParams.d_value,
            ModelHyperParams.d_model, ModelHyperParams.d_inner_hid,
            ModelHyperParams.prepostprocess_dropout,
            ModelHyperParams.attention_dropout, ModelHyperParams.relu_dropout,
            ModelHyperParams.preprocess_cmd, ModelHyperParams.postprocess_cmd,
            ModelHyperParams.weight_sharing, TrainTaskConfig.label_smooth_eps)
        # define optimizer
        optimizer = fluid.optimizer.Adam(learning_rate=NoamDecay(
            ModelHyperParams.d_model, TrainTaskConfig.warmup_steps,
            TrainTaskConfig.learning_rate),
                                         beta1=TrainTaskConfig.beta1,
                                         beta2=TrainTaskConfig.beta2,
                                         epsilon=TrainTaskConfig.eps)
        #
        if args.use_data_parallel:
            transformer = fluid.dygraph.parallel.DataParallel(
                transformer, strategy)

        # define data generator for training and validation
        train_reader = paddle.batch(wmt16.train(
            ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size),
                                    batch_size=TrainTaskConfig.batch_size)
        if args.use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)
        val_reader = paddle.batch(wmt16.test(ModelHyperParams.src_vocab_size,
                                             ModelHyperParams.trg_vocab_size),
                                  batch_size=TrainTaskConfig.batch_size)

        # loop for training iterations
        for i in range(TrainTaskConfig.pass_num):
            dy_step = 0
            sum_cost = 0
            transformer.train()
            for batch in train_reader():
                enc_inputs, dec_inputs, label, weights = prepare_train_input(
                    batch, ModelHyperParams.eos_idx, ModelHyperParams.eos_idx,
                    ModelHyperParams.n_head)

                dy_sum_cost, dy_avg_cost, dy_predict, dy_token_num = transformer(
                    enc_inputs, dec_inputs, label, weights)

                if args.use_data_parallel:
                    dy_avg_cost = transformer.scale_loss(dy_avg_cost)
                    dy_avg_cost.backward()
                    transformer.apply_collective_grads()
                else:
                    dy_avg_cost.backward()
                optimizer.minimize(dy_avg_cost)
                transformer.clear_gradients()

                dy_step = dy_step + 1
                if dy_step % 10 == 0:
                    print("pass num : {}, batch_id: {}, dy_graph avg loss: {}".
                          format(i, dy_step,
                                 dy_avg_cost.numpy() * trainer_count))

            # switch to evaluation mode
            transformer.eval()
            sum_cost = 0
            token_num = 0
            for batch in val_reader():
                enc_inputs, dec_inputs, label, weights = prepare_train_input(
                    batch, ModelHyperParams.eos_idx, ModelHyperParams.eos_idx,
                    ModelHyperParams.n_head)

                dy_sum_cost, dy_avg_cost, dy_predict, dy_token_num = transformer(
                    enc_inputs, dec_inputs, label, weights)
                sum_cost += dy_sum_cost.numpy()
                token_num += dy_token_num.numpy()
            print("pass : {} finished, validation avg loss: {}".format(
                i, sum_cost / token_num))

        if fluid.dygraph.parallel.Env().dev_id == 0:
            fluid.save_dygraph(transformer.state_dict(), args.model_file)


if __name__ == '__main__':
    args = parse_args()
    train(args)
