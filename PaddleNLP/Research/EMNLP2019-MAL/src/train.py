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
# limitations under the License.

import argparse
import ast
import copy
import logging
import multiprocessing
import os
import six
import sys
import time
import random
import math

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.layers.nn as nn
import paddle.fluid.layers.tensor as tensor
from paddle.fluid.framework import default_main_program

import reader
from reader import *
from config import *
from forward_model import forward_transformer, forward_position_encoding_init, forward_fast_decode, make_all_py_reader_inputs
from dense_model import dense_transformer, dense_fast_decode
from relative_model import relative_transformer, relative_fast_decode

def parse_args():
    """
        parse_args
    """
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--train_file_pattern",
        type=str,
        required=True,
        help="The pattern to match training data files.")
    parser.add_argument(
        "--val_file_pattern",
        type=str,
        help="The pattern to match validation data files.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="The pattern to match training data files.")
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        help="Infer batch_size")
    parser.add_argument(
        "--decode_alpha",
        type=float,
        help="decode_alpha")
    parser.add_argument(
        "--beam_size",
        type=int,
        help="Infer beam_size")
    parser.add_argument(
        "--use_token_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to "
        "produce batch data according to token number.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=200000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=2,
        help="The number of threads which executor use.")
    parser.add_argument(
        "--use_fp16",
        type=ast.literal_eval,
        default=True,
        help="Use fp16 or not"
    )

    parser.add_argument(
        "--nccl_comm_num",
        type=int,
        default=1,
        help="The number of threads which executor use.")

    parser.add_argument(
        "--sort_type",
        default="pool",
        choices=("global", "pool", "none"),
        help="The grain to sort by length: global for all instances; pool for "
        "instances in pool; none for no sort.")
    parser.add_argument(
        "--use_hierarchical_allreduce",
        default=False,
        type=ast.literal_eval,
        help="Use hierarchical allreduce or not.")
    parser.add_argument(
        "--hierarchical_allreduce_inter_nranks",
        default=8,
        type=int,
        help="interranks.")
    parser.add_argument(
        "--shuffle",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument(
        "--shuffle_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle the data batches.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        "--token_delimiter",
        type=lambda x: str(x.encode().decode("unicode-escape")),
        default=" ",
        help="The delimiter used to split tokens in source or target sentences. "
        "For EN-DE BPE data we provided, use spaces as token delimiter. ")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--local',
        type=ast.literal_eval,
        default=False,
        help='Whether to run as local mode.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help="The device type.")
    parser.add_argument(
        '--sync', type=ast.literal_eval, default=True, help="sync mode.")
    parser.add_argument(
        "--enable_ce",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to run the task "
        "for continuous evaluation.")
    parser.add_argument(
        "--use_mem_opt",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to use memory optimization.")
    parser.add_argument(
        "--use_py_reader",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to use py_reader.")
    parser.add_argument(
        "--fetch_steps",
        type=int,
        default=100,
        help="The frequency to fetch and print output.")
    parser.add_argument(
        "--use_delay_load",
        type=ast.literal_eval,
        default=True,
        help=
        "The flag indicating whether to load all data into memories at once.")
    parser.add_argument(
        "--src_vocab_size",
        type=str,
        required=True,
        help="Size of src Vocab.")
    parser.add_argument(
        "--tgt_vocab_size",
        type=str,
        required=True,
        help="Size of tgt Vocab.")
    parser.add_argument(
        "--restore_step",
        type=int,
        default=0,
        help="The step number of checkpoint to restore training.")
    parser.add_argument(
        "--fuse",
        type=int,
        default=0,
        help="Use fusion or not.")

    args = parser.parse_args()

    src_voc_size = args.src_vocab_size
    trg_voc_size = args.tgt_vocab_size
    if args.use_delay_load:
        dict_args = [
            "src_vocab_size", src_voc_size,
            "trg_vocab_size", trg_voc_size,
            "bos_idx", str(0),
            "eos_idx", str(1),
            "unk_idx", str(int(src_voc_size) - 1)
        ]
    else:
        src_dict = reader.DataReader.load_dict(args.src_vocab_fpath)
        trg_dict = reader.DataReader.load_dict(args.trg_vocab_fpath)
        dict_args = [
            "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
            str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
            "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
            str(src_dict[args.special_token[2]])
        ]
    merge_cfg_from_list(args.opts + dict_args,
                        [TrainTaskConfig, ModelHyperParams])
    return args


def prepare_batch_input(insts, data_input_names, src_pad_idx, trg_pad_idx,
                        n_head, d_model):
    """
    Put all padded data needed by training into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)

    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [inst[1] for inst in insts], trg_pad_idx, n_head, is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len, 1)
    trg_word = trg_word[:, 1:, :]
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

    # reverse_target
    reverse_trg_word, _, _, _ = pad_batch_data(
        [inst[3] for inst in insts], trg_pad_idx, n_head, is_target=True)
    reverse_trg_word = reverse_trg_word.reshape(-1, trg_max_len, 1)
    reverse_trg_word = reverse_trg_word[:, 1:, :]

    reverse_lbl_word, _, _ = pad_batch_data(
        [inst[4] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)

    eos_position = []
    meet_eos = False
    for word_id in reverse_lbl_word:
        if word_id[0] == 1 and not meet_eos:
            meet_eos = True
            eos_position.append([1])
        elif word_id[0] == 1 and meet_eos:
            eos_position.append([0])
        else:
            meet_eos = False
            eos_position.append([0])

    data_input_dict = dict(
        zip(data_input_names, [
            src_word, src_pos, src_slf_attn_bias, trg_word, reverse_trg_word, trg_pos,
            trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight, reverse_lbl_word, np.asarray(eos_position, dtype = "int64")
        ]))

    return data_input_dict, np.asarray([num_token], dtype="float32")


def prepare_feed_dict_list(data_generator, count, num_tokens=None, num_insts=None):
    """
    Prepare the list of feed dict for multi-devices.
    """
    feed_dict_list = []
    eos_idx = ModelHyperParams.eos_idx
    n_head =  ModelHyperParams.n_head
    d_model = ModelHyperParams.d_model
    max_length = ModelHyperParams.max_length
    dense_n_head = DenseModelHyperParams.n_head
    dense_d_model = DenseModelHyperParams.d_model

    if data_generator is not None:  # use_py_reader == False
        dense_data_input_names = dense_encoder_data_input_fields + \
                        dense_decoder_data_input_fields[:-1] + dense_label_data_input_fields
        data_input_names = encoder_data_input_fields + \
                        decoder_data_input_fields[:-1] + label_data_input_fields
        data = next(data_generator)
        for idx, data_buffer in enumerate(data):
            data_input_dict, num_token = prepare_batch_input(
                data_buffer, data_input_names, eos_idx,
                eos_idx, n_head,
                d_model)
            dense_data_input_dict, _ = prepare_batch_input(
                data_buffer, dense_data_input_names, eos_idx,
                eos_idx, dense_n_head,
                dense_d_model)
            data_input_dict.update(dense_data_input_dict) # merge dict
            feed_dict_list.append(data_input_dict)
            if isinstance(num_tokens, list): num_tokens.append(num_token)
            if isinstance(num_insts, list): num_insts.append(len(data_buffer))

    return feed_dict_list if len(feed_dict_list) == count else None


def py_reader_provider_wrapper(data_reader):
    """
    Data provider needed by fluid.layers.py_reader.
    """

    def py_reader_provider():
        """
            py_reader_provider
        """
        eos_idx = ModelHyperParams.eos_idx
        n_head =  ModelHyperParams.n_head
        d_model = ModelHyperParams.d_model
        max_length = ModelHyperParams.max_length
        dense_n_head = DenseModelHyperParams.n_head
        dense_d_model = DenseModelHyperParams.d_model

        data_input_names = encoder_data_input_fields + \
                    decoder_data_input_fields[:-1] + label_data_input_fields
        dense_data_input_names = dense_encoder_data_input_fields + \
                    dense_decoder_data_input_fields[:-1] + label_data_input_fields

        new_data_input_names = data_input_names + dense_bias_input_fields

        for batch_id, data in enumerate(data_reader()):
            data_input_dict, num_token = prepare_batch_input(
                data, data_input_names, eos_idx,
                eos_idx, n_head,
                d_model)
            dense_data_input_dict, _ = prepare_batch_input(
                data, dense_data_input_names, eos_idx,
                eos_idx, dense_n_head,
                dense_d_model)
            data_input_dict["dense_src_slf_attn_bias"] = dense_data_input_dict["dense_src_slf_attn_bias"]
            data_input_dict["dense_trg_slf_attn_bias"] = dense_data_input_dict["dense_trg_slf_attn_bias"]
            data_input_dict["dense_trg_src_attn_bias"] = dense_data_input_dict["dense_trg_src_attn_bias"]
            total_dict = dict(data_input_dict.items())
            yield [total_dict[item] for item in new_data_input_names]

    return py_reader_provider


from infer import prepare_feed_dict_list as infer_prepare_feed_dict_list
from infer import prepare_dense_feed_dict_list as infer_prepare_dense_feed_dict_list
def test_context(exe, train_exe, dev_count, agent_name, args):
    # Context to do validation.
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    if args.enable_ce:
        test_prog.random_seed = 1000
        startup_prog.random_seed = 1000
    with fluid.program_guard(test_prog, startup_prog):
        if agent_name == "new_forward":
            with fluid.unique_name.guard("new_forward"):
                out_ids1, out_scores1 = forward_fast_decode(
                    ModelHyperParams.src_vocab_size,
                    ModelHyperParams.trg_vocab_size,
                    ModelHyperParams.max_length + 50,
                    ModelHyperParams.n_layer,
                    ModelHyperParams.n_head,
                    ModelHyperParams.d_key,
                    ModelHyperParams.d_value,
                    ModelHyperParams.d_model,
                    ModelHyperParams.d_inner_hid,
                    ModelHyperParams.prepostprocess_dropout,
                    ModelHyperParams.attention_dropout,
                    ModelHyperParams.relu_dropout,
                    ModelHyperParams.preprocess_cmd,
                    ModelHyperParams.postprocess_cmd,
                    ModelHyperParams.weight_sharing,
                    ModelHyperParams.embedding_sharing,
                    args.beam_size,
                    args.infer_batch_size,
                    InferTaskConfig.max_out_len,
                    args.decode_alpha,
                    ModelHyperParams.eos_idx,
                    params_type="new"
                    )
        elif agent_name == "new_relative_position":
            with fluid.unique_name.guard("new_relative_position"):
                out_ids2, out_scores2 = relative_fast_decode(
                    ModelHyperParams.src_vocab_size,
                    ModelHyperParams.trg_vocab_size,
                    ModelHyperParams.max_length + 50,
                    ModelHyperParams.n_layer,
                    ModelHyperParams.n_head,
                    ModelHyperParams.d_key,
                    ModelHyperParams.d_value,
                    ModelHyperParams.d_model,
                    ModelHyperParams.d_inner_hid,
                    ModelHyperParams.prepostprocess_dropout,
                    ModelHyperParams.attention_dropout,
                    ModelHyperParams.relu_dropout,
                    ModelHyperParams.preprocess_cmd,
                    ModelHyperParams.postprocess_cmd,
                    ModelHyperParams.weight_sharing,
                    ModelHyperParams.embedding_sharing,
                    args.beam_size,
                    args.infer_batch_size,
                    InferTaskConfig.max_out_len,
                    args.decode_alpha,
                    ModelHyperParams.eos_idx,
                    params_type="new"
                    )

        elif agent_name == "new_dense":
            DenseModelHyperParams.src_vocab_size = ModelHyperParams.src_vocab_size
            DenseModelHyperParams.trg_vocab_size = ModelHyperParams.trg_vocab_size
            DenseModelHyperParams.weight_sharing = ModelHyperParams.weight_sharing
            DenseModelHyperParams.embedding_sharing = ModelHyperParams.embedding_sharing 
            with fluid.unique_name.guard("new_dense"):
                out_ids3, out_scores3 = dense_fast_decode(
                    DenseModelHyperParams.src_vocab_size,
                    DenseModelHyperParams.trg_vocab_size,
                    DenseModelHyperParams.max_length + 50,
                    DenseModelHyperParams.n_layer,
                    DenseModelHyperParams.enc_n_layer,
                    DenseModelHyperParams.n_head,
                    DenseModelHyperParams.d_key,
                    DenseModelHyperParams.d_value,
                    DenseModelHyperParams.d_model,
                    DenseModelHyperParams.d_inner_hid,
                    DenseModelHyperParams.prepostprocess_dropout,
                    DenseModelHyperParams.attention_dropout,
                    DenseModelHyperParams.relu_dropout,
                    DenseModelHyperParams.preprocess_cmd,
                    DenseModelHyperParams.postprocess_cmd,
                    DenseModelHyperParams.weight_sharing,
                    DenseModelHyperParams.embedding_sharing,
                    args.beam_size,
                    args.infer_batch_size,
                    InferTaskConfig.max_out_len,
                    args.decode_alpha,
                    ModelHyperParams.eos_idx,
                    params_type="new"
                    )

    test_prog = test_prog.clone(for_test=True)

    dev_count = 1
    file_pattern = "%s" % (args.val_file_pattern)
    lines_cnt = len(open(file_pattern, 'r').readlines())
    data_reader = line_reader(file_pattern, args.infer_batch_size, dev_count,
                    token_delimiter=args.token_delimiter,
                    max_len=ModelHyperParams.max_length,
                    parse_line=parse_src_line)

    test_data = prepare_data_generator(args, is_test=True, count=dev_count, pyreader=None,
                                       batch_size=args.infer_batch_size, data_reader=data_reader)

    def test(step_id, exe=exe):

        f = ""
        if agent_name == "new_relative_position":
            f = open("./output/new_relative_position_iter_%d.trans" % (step_id), 'w')
        elif agent_name == "new_forward":
            f = open("./output/new_forward_iter_%d.trans" % (step_id), 'w')
        elif agent_name == "new_dense":
            f = open("./output/new_dense_iter_%d.trans" % (step_id), 'w')

        data_generator = test_data()
        trans_list = []
        while True:
            try:
                feed_dict_list = infer_prepare_feed_dict_list(data_generator, 1) if agent_name != "new_dense" else infer_prepare_dense_feed_dict_list(data_generator, 1)
                if agent_name == "new_forward":
                    seq_ids, seq_scores = exe.run(
                            fetch_list=[out_ids1.name, out_scores1.name],
                            feed=feed_dict_list,
                            program=test_prog,
                            return_numpy=True)
                elif agent_name == "new_relative_position":
                    seq_ids, seq_scores = exe.run(
                            fetch_list=[out_ids2.name, out_scores2.name],
                            feed=feed_dict_list,
                            program=test_prog,
                            return_numpy=True)
                elif agent_name == "new_dense":
                    seq_ids, seq_scores = exe.run(
                            fetch_list=[out_ids3.name, out_scores3.name],
                            feed=feed_dict_list,
                            program=test_prog,
                            return_numpy=True)

                seq_ids = seq_ids.tolist()
                for index in xrange(args.infer_batch_size):
                    seq = seq_ids[index][0]
                    if 1 not in seq:
                        res = seq[1:-1]
                    else:
                        res = seq[1: seq.index(1)]
                    res = map(str, res)
                    trans_list.append(" ".join(res))
            except (StopIteration, fluid.core.EOFException):
                # The current pass is over.
                break
        trans_list = trans_list[:lines_cnt]
        for trans in trans_list:
            f.write("%s\n" % trans)

        f.flush()
        f.close()
    return test

def get_tensor_by_prefix(pre_name, param_name_list):
    tensors_list = []
    for param_name in param_name_list:
        if pre_name in param_name:
            tensors_list.append(fluid.global_scope().find_var(param_name).get_tensor())

    if pre_name == "fixed_relative_positionfixed_relative_position":
        tensors_list.append(fluid.global_scope().find_var("fixed_relative_positions_keys").get_tensor())
        tensors_list.append(fluid.global_scope().find_var("fixed_relative_positions_values").get_tensor())
    elif pre_name == "new_relative_positionnew_relative_position":
        tensors_list.append(fluid.global_scope().find_var("new_relative_positions_keys").get_tensor())
        tensors_list.append(fluid.global_scope().find_var("new_relative_positions_values").get_tensor())

    return tensors_list


def train_loop(exe,
               train_prog,
               startup_prog,
               args,
               dev_count,
               avg_cost,
               teacher_cost,
               single_model_sum_cost,
               single_model_avg_cost,
               token_num,
               pyreader, place,
               nccl2_num_trainers=1,
               nccl2_trainer_id=0,
               scaled_cost=None,
               loss_scaling=None
               ):
    """
        train_loop
    """
    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        exe.run(startup_prog)
        logging.info("load checkpoint from {}".format(TrainTaskConfig.ckpt_path))
        fluid.io.load_params(exe, TrainTaskConfig.ckpt_path, main_program=train_prog)
    else:
        logging.info("init fluid.framework.default_startup_program")
        exe.run(startup_prog)

    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list ]

    logging.info("begin reader")
    batch_scheme = batching_scheme(args.batch_size, 256, shard_multiplier=nccl2_num_trainers)
    tf_data = bucket_by_sequence_length(
                repeat(
                        interleave_reader(
                            args.train_file_pattern,
                            cycle_length=8,
                            token_delimiter=args.token_delimiter,
                            max_len=ModelHyperParams.max_length,
                            parse_line=parse_line,
                        ), -1),
                    lambda x:max(len(x[0]), len(x[1])),
                    batch_scheme["boundaries"],
                    batch_scheme["batch_sizes"],
                    nccl2_num_trainers,
                    nccl2_trainer_id
    )
    args.use_token_batch = False
    train_data = prepare_data_generator(
        args, is_test=False, count=dev_count, pyreader=pyreader, data_reader=tf_data, \
                py_reader_provider_wrapper=py_reader_provider_wrapper)

    # For faster executor
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 20
    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.fuse_all_optimizer_ops = False
    build_strategy.fuse_all_reduce_ops = False
    build_strategy.enable_backward_optimizer_op_deps = True
    if args.fuse:
        build_strategy.fuse_all_reduce_ops = True


    trainer_id = nccl2_trainer_id
    train_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        loss_name=avg_cost.name,
        main_program=train_prog,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy,
        num_trainers=nccl2_num_trainers,
        trainer_id=nccl2_trainer_id)

    if args.val_file_pattern is not None:
        new_forward_test = test_context(exe, train_exe, dev_count, "new_forward", args)
        new_dense_test = test_context(exe, train_exe, dev_count, "new_dense", args)
        new_relative_position_test = test_context(exe, train_exe, dev_count, "new_relative_position", args)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -((1. - TrainTaskConfig.label_smooth_eps) * np.log(
        (1. - TrainTaskConfig.label_smooth_eps
         )) + TrainTaskConfig.label_smooth_eps *
                        np.log(TrainTaskConfig.label_smooth_eps / (
                            ModelHyperParams.trg_vocab_size - 1) + 1e-20))

    # set recovery step
    step_idx = args.restore_step if args.restore_step else 0
    if step_idx != 0:
        var = fluid.global_scope().find_var("@LR_DECAY_COUNTER@").get_tensor()
        recovery_step = np.array([step_idx]).astype("int64")
        var.set(recovery_step, fluid.CPUPlace())
        step = np.array(var)[0]


    # set pos encoding
    model_prefix = ["fixed_forward", "fixed_relative_position",
                    "new_forward", "new_relative_position"]
    for pos_enc_param_name in pos_enc_param_names:
        for prefix in model_prefix:
            pos_name = prefix * 2 + pos_enc_param_name
            pos_enc_param = fluid.global_scope().find_var(
                pos_name).get_tensor()

            pos_enc_param.set(
                forward_position_encoding_init(
                        ModelHyperParams.max_length + 50, 
                        ModelHyperParams.d_model), place)

    model_prefix_2 = ["fixed_dense", "new_dense"]
    for pos_enc_param_name in pos_enc_param_names:
        for prefix in model_prefix_2:
            pos_name = prefix * 2 + pos_enc_param_name
            pos_enc_param = fluid.global_scope().find_var(
                pos_name).get_tensor()
            
            pos_enc_param.set(
                forward_position_encoding_init(
                        DenseModelHyperParams.max_length + 50,
                        DenseModelHyperParams.d_model), place)
    

    logging.info("begin train")
    for pass_id in six.moves.xrange(TrainTaskConfig.pass_num):
        pass_start_time = time.time()
        avg_batch_time = time.time()

        pyreader.start()
        data_generator = None

        batch_id = 0
        while True:
            try:
                num_tokens = []
                num_insts = []
                feed_dict_list = prepare_feed_dict_list(data_generator,
                                                        dev_count, num_tokens, num_insts)

                num_token = np.sum(num_tokens).reshape([-1])
                num_inst = np.sum(num_insts).reshape([-1])

                outs = train_exe.run(
                    fetch_list=[avg_cost.name, token_num.name, teacher_cost.name]
                    if (step_idx == 0 or step_idx % args.fetch_steps == (args.fetch_steps - 1)) else [],
                    feed=feed_dict_list)

                if (step_idx == 0 or step_idx % args.fetch_steps == (args.fetch_steps - 1)):
                    single_model_total_avg_cost, token_num_val = np.array(outs[0]), np.array(outs[1])
                    teacher = np.array(outs[2])

                    if step_idx == 0:
                        logging.info(
                            ("step_idx: %d, epoch: %d, batch: %d, teacher loss: %f, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" + (", batch size: %d" if num_inst else "")) %
                            ((step_idx, pass_id, batch_id, teacher, single_model_total_avg_cost,
                             single_model_total_avg_cost - loss_normalizer,
                             np.exp([min(single_model_total_avg_cost, 100)])) + ((num_inst,) if num_inst else ())))
                    else:
                        logging.info(
                            ("step_idx: %d, epoch: %d, batch: %d, teacher loss: %f, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, speed: %.2f step/s" + \
                            (", batch size: %d" if num_inst else "")) %
                            ((step_idx, pass_id, batch_id, teacher, single_model_total_avg_cost,
                             single_model_total_avg_cost - loss_normalizer,
                             np.exp([min(single_model_total_avg_cost, 100)]),
                             args.fetch_steps / (time.time() - avg_batch_time)) + ((num_inst,) if num_inst else ())))
                        avg_batch_time = time.time()

                if step_idx % TrainTaskConfig.fixed_freq == (TrainTaskConfig.fixed_freq - 1):
                    logging.info("copy parameters to fixed parameters when step_idx is {}".format(step_idx))

                    fixed_forward_tensors = get_tensor_by_prefix("fixed_forwardfixed_forward", param_name_list)
                    new_forward_tensors = get_tensor_by_prefix("new_forwardnew_forward", param_name_list)
                    fixed_dense_tensors = get_tensor_by_prefix("fixed_densefixed_dense", param_name_list)
                    new_dense_tensors = get_tensor_by_prefix("new_densenew_dense", param_name_list)
                    fixed_relative_tensors = get_tensor_by_prefix("fixed_relative_positionfixed_relative_position", param_name_list)
                    new_relative_tensors = get_tensor_by_prefix("new_relative_positionnew_relative_position", param_name_list)

                    for (fixed_tensor, new_tensor) in zip(fixed_forward_tensors, new_forward_tensors):
                        fixed_tensor.set(np.array(new_tensor), place)
                    for (fixed_tensor, new_tensor) in zip(fixed_relative_tensors, new_relative_tensors):
                        fixed_tensor.set(np.array(new_tensor), place)
                    for (fixed_tensor, new_tensor) in zip(fixed_dense_tensors, new_dense_tensors):
                        fixed_tensor.set(np.array(new_tensor), place)

                if step_idx % TrainTaskConfig.save_freq == (TrainTaskConfig.save_freq - 1):
                    if trainer_id == 0:
                        fluid.io.save_params(
                            exe,
                            os.path.join(TrainTaskConfig.model_dir,
                                         "iter_" + str(step_idx) + ".infer.model"),train_prog)

                        if args.val_file_pattern is not None:
                            train_exe.drop_local_exe_scopes()
                            new_dense_test(step_idx)
                            new_forward_test(step_idx)
                            new_relative_position_test(step_idx)

                batch_id += 1
                step_idx += 1
            except (StopIteration, fluid.core.EOFException):
                break


def train(args):
    """
        train
    """
    is_local = os.getenv("PADDLE_IS_LOCAL", "1")
    if is_local == '0':
        args.local = False
    print(args)

    if args.device == 'CPU':
        TrainTaskConfig.use_gpu = False

    training_role = os.getenv("TRAINING_ROLE", "TRAINER")
    gpus = os.getenv("FLAGS_selected_gpus").split(",")
    gpu_id = int(gpus[0])

    if training_role == "PSERVER" or (not TrainTaskConfig.use_gpu):
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        place = fluid.CUDAPlace(gpu_id)
        dev_count = len(gpus)

    exe = fluid.Executor(place)

    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    if args.enable_ce:
        train_prog.random_seed = 1000
        startup_prog.random_seed = 1000

    with fluid.program_guard(train_prog, startup_prog):
        logits_list = []

        data_input_names = encoder_data_input_fields + \
                decoder_data_input_fields[:-1] + label_data_input_fields + dense_bias_input_fields

        all_data_inputs, pyreader = make_all_py_reader_inputs(data_input_names)
        with fluid.unique_name.guard("new_forward"):
            new_forward_sum_cost, new_forward_avg_cost, new_forward_token_num, new_forward_logits, new_forward_xent, new_forward_loss, new_forward_label, new_forward_non_zeros = forward_transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 50,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                ModelHyperParams.embedding_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=True,
                is_test=False,
                params_type="new",
                all_data_inputs=all_data_inputs)

        with fluid.unique_name.guard("new_relative_position"):
            new_relative_position_sum_cost, new_relative_position_avg_cost, new_relative_position_token_num, new_relative_position_logits, new_relative_position_xent, new_relative_position_loss, new_relative_position_label, new_relative_position_non_zeros = relative_transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 50,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                ModelHyperParams.embedding_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=False,
                params_type="new",
                all_data_inputs=all_data_inputs)

        DenseModelHyperParams.src_vocab_size = ModelHyperParams.src_vocab_size
        DenseModelHyperParams.trg_vocab_size = ModelHyperParams.trg_vocab_size
        DenseModelHyperParams.weight_sharing = ModelHyperParams.weight_sharing
        DenseModelHyperParams.embedding_sharing = ModelHyperParams.embedding_sharing

        with fluid.unique_name.guard("new_dense"):
            new_dense_sum_cost, new_dense_avg_cost, new_dense_token_num, new_dense_logits, new_dense_xent, new_dense_loss, new_dense_label, _ = dense_transformer(
                DenseModelHyperParams.src_vocab_size,
                DenseModelHyperParams.trg_vocab_size,
                DenseModelHyperParams.max_length + 50,
                DenseModelHyperParams.n_layer,
                DenseModelHyperParams.enc_n_layer,
                DenseModelHyperParams.n_head,
                DenseModelHyperParams.d_key,
                DenseModelHyperParams.d_value,
                DenseModelHyperParams.d_model,
                DenseModelHyperParams.d_inner_hid,
                DenseModelHyperParams.prepostprocess_dropout,
                DenseModelHyperParams.attention_dropout,
                DenseModelHyperParams.relu_dropout,
                DenseModelHyperParams.preprocess_cmd,
                DenseModelHyperParams.postprocess_cmd,
                DenseModelHyperParams.weight_sharing,
                DenseModelHyperParams.embedding_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=False,
                params_type="new",
                all_data_inputs=all_data_inputs)

        with fluid.unique_name.guard("fixed_forward"):
            fixed_forward_sum_cost, fixed_forward_avg_cost, fixed_forward_token_num, fixed_forward_logits, fixed_forward_xent, fixed_forward_loss, fixed_forward_label, fixed_forward_non_zeros = forward_transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 50,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                ModelHyperParams.embedding_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=False,
                params_type="fixed",
                all_data_inputs=all_data_inputs)
            logits_list.append(fixed_forward_logits)


        DenseModelHyperParams.src_vocab_size = ModelHyperParams.src_vocab_size
        DenseModelHyperParams.trg_vocab_size = ModelHyperParams.trg_vocab_size
        DenseModelHyperParams.weight_sharing = ModelHyperParams.weight_sharing
        DenseModelHyperParams.embedding_sharing = ModelHyperParams.embedding_sharing

        with fluid.unique_name.guard("fixed_dense"):
            fixed_dense_sum_cost, fixed_dense_avg_cost, fixed_dense_token_num, fixed_dense_logits, fixed_dense_xent, fixed_dense_loss, fixed_dense_label, _ = dense_transformer(
                DenseModelHyperParams.src_vocab_size,
                DenseModelHyperParams.trg_vocab_size,
                DenseModelHyperParams.max_length + 50,
                DenseModelHyperParams.n_layer,
                DenseModelHyperParams.enc_n_layer,
                DenseModelHyperParams.n_head,
                DenseModelHyperParams.d_key,
                DenseModelHyperParams.d_value,
                DenseModelHyperParams.d_model,
                DenseModelHyperParams.d_inner_hid,
                DenseModelHyperParams.prepostprocess_dropout,
                DenseModelHyperParams.attention_dropout,
                DenseModelHyperParams.relu_dropout,
                DenseModelHyperParams.preprocess_cmd,
                DenseModelHyperParams.postprocess_cmd,
                DenseModelHyperParams.weight_sharing,
                DenseModelHyperParams.embedding_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=False,
                params_type="fixed",
                all_data_inputs=all_data_inputs)
            logits_list.append(fixed_dense_logits)

        with fluid.unique_name.guard("fixed_relative_position"):
            fixed_relative_sum_cost, fixed_relative_avg_cost, fixed_relative_token_num, fixed_relative_logits, fixed_relative_xent, fixed_relative_loss, fixed_relative_label, _ = relative_transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 50,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                ModelHyperParams.embedding_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=False,
                params_type="fixed",
                all_data_inputs=all_data_inputs)
            logits_list.append(fixed_relative_logits)

        # normalizing
        confidence = 1.0 - TrainTaskConfig.label_smooth_eps
        low_confidence = (1.0 - confidence) / (ModelHyperParams.trg_vocab_size - 1)
        normalizing = -(confidence * math.log(confidence) + (ModelHyperParams.trg_vocab_size - 1) *
                low_confidence * math.log(low_confidence + 1e-20))

        batch_size = layers.shape(new_forward_logits)[0]
        seq_length = layers.shape(new_forward_logits)[1]
        trg_voc_size = layers.shape(new_forward_logits)[2]
        
        # ensemble
        teacher_logits = logits_list[0]
        for index in xrange(1, len(logits_list)):
            teacher_logits += logits_list[index]
        
        teacher_logits = teacher_logits / len(logits_list)

        # new_target
        new_target = layers.softmax(teacher_logits)
        new_target.stop_gradient = True

        # agent_1: forward
        fdistill_xent = layers.softmax_with_cross_entropy(
                logits=new_forward_logits,
                label=new_target,
                soft_label=True)
        fdistill_xent -= normalizing
        fdistill_loss = layers.reduce_sum(fdistill_xent * new_forward_non_zeros) / new_forward_token_num

         # agent_2: relative
        rdistill_xent = layers.softmax_with_cross_entropy(
                logits=new_relative_position_logits,
                label=new_target,
                soft_label=True)
        rdistill_xent -= normalizing
        rdistill_loss = layers.reduce_sum(rdistill_xent * new_forward_non_zeros) / new_forward_token_num

        # agent_3: dense
        ddistill_xent = layers.softmax_with_cross_entropy(
                logits=new_dense_logits,
                label=new_target,
                soft_label=True)
        ddistill_xent -= normalizing
        ddistill_loss = layers.reduce_sum(ddistill_xent * new_forward_non_zeros) / new_forward_token_num

        
        teacher_loss = fixed_forward_avg_cost + fixed_dense_avg_cost + fixed_relative_avg_cost
        avg_cost = TrainTaskConfig.beta * new_forward_avg_cost + (1.0 - TrainTaskConfig.beta) * fdistill_loss + TrainTaskConfig.beta * new_relative_position_avg_cost + (1.0 - TrainTaskConfig.beta) * rdistill_loss + TrainTaskConfig.beta * new_dense_avg_cost + (1.0 - TrainTaskConfig.beta) * ddistill_loss + teacher_loss


        avg_cost.persistable = True
        teacher_loss.persistable = True

        optimizer = None
        if args.sync:
            lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
                ModelHyperParams.d_model, TrainTaskConfig.warmup_steps)
            logging.info("before adam")

            with fluid.default_main_program()._lr_schedule_guard():
                learning_rate = lr_decay * TrainTaskConfig.learning_rate
            optimizer = fluid.optimizer.Adam(
                learning_rate=learning_rate,
                beta1=TrainTaskConfig.beta1,
                beta2=TrainTaskConfig.beta2,
                epsilon=TrainTaskConfig.eps)
        else:
            optimizer = fluid.optimizer.SGD(0.003)
        if args.use_fp16:
            #black_varnames={"src_slf_attn_bias", "trg_slf_attn_bias", "trg_src_attn_bias", "dense_src_slf_attn_bias", "dense_trg_slf_attn_bias", "dense_trg_src_attn_bias"}
            #amp_lists=fluid.contrib.mixed_precision.AutoMixedPrecisionLists(custom_black_varnames=black_varnames,
            #        custom_black_list=["dropout"])
            #optimizer = fluid.contrib.mixed_precision.decorate(optimizer, amp_lists=amp_lists,
            optimizer = fluid.contrib.mixed_precision.decorate(optimizer,
                                                                init_loss_scaling=32768, incr_every_n_steps=2000,
                                                                use_dynamic_loss_scaling=True)

        optimizer.minimize(avg_cost)

        loss_scaling=None
        scaled_cost=None
        if args.use_fp16:
            scaled_cost = optimizer.get_scaled_loss()
            loss_scaling = optimizer.get_loss_scaling()

    if args.local:
        logging.info("local start_up:")
        train_loop(exe, train_prog, startup_prog, args, dev_count, avg_cost, teacher_loss, new_relative_position_sum_cost, new_relative_position_avg_cost,
                   new_relative_position_token_num, pyreader, place)
    else:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        worker_endpoints = worker_endpoints_env.split(",")
        trainers_num = len(worker_endpoints)

        logging.info("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
                trainer_id:{}".format(worker_endpoints, trainers_num,
                                      current_endpoint, trainer_id))

        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        if args.nccl_comm_num > 1:
            config.nccl_comm_num = args.nccl_comm_num
        if args.use_hierarchical_allreduce and trainers_num > args.hierarchical_allreduce_inter_nranks:
            logging.info("use_hierarchical_allreduce")
            config.use_hierarchical_allreduce=args.use_hierarchical_allreduce

            config.hierarchical_allreduce_inter_nranks=8
            if config.hierarchical_allreduce_inter_nranks > 1:
                config.hierarchical_allreduce_inter_nranks=args.hierarchical_allreduce_inter_nranks

            assert config.hierarchical_allreduce_inter_nranks > 1
            assert trainers_num % config.hierarchical_allreduce_inter_nranks == 0

            config.hierarchical_allreduce_exter_nranks = \
                trainers_num / config.hierarchical_allreduce_inter_nranks

        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id, trainers=worker_endpoints_env,
            current_endpoint=current_endpoint, program=train_prog,
            startup_program=startup_prog)

        train_loop(exe, train_prog, startup_prog, args, dev_count, avg_cost, teacher_loss,
                   new_relative_position_sum_cost, new_relative_position_avg_cost, new_relative_position_token_num, pyreader, place, trainers_num, trainer_id, scaled_cost=scaled_cost, loss_scaling=loss_scaling)

        
if __name__ == "__main__":
    LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG, format=LOG_FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()
    train(args)
