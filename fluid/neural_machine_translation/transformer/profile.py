import os
import time
import argparse
import ast
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

from train import split_data, read_multiple, prepare_batch_input
from model import transformer, position_encoding_init
from optim import LearningRateScheduler
from config import *
import reader


def parse_args():
    parser = argparse.ArgumentParser(
        "Profile the training process for Transformer.")
    parser.add_argument(
        "--src_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of source language.")
    parser.add_argument(
        "--trg_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of target language.")
    parser.add_argument(
        "--train_file_pattern",
        type=str,
        required=True,
        help="The pattern to match training data files.")
    parser.add_argument(
        "--use_token_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to "
        "produce batch data according to token number.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=10,
        help="The maximum number of iterations profiling over.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=10000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help="The device type.")

    args = parser.parse_args()
    # Append args related to dict
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


def train_loop(exe, train_progm, init, num_iters, train_data, dev_count,
               sum_cost, avg_cost, lr_scheduler, token_num, predict):

    data_input_names = encoder_data_input_fields + decoder_data_input_fields[:
                                                                             -1] + label_data_input_fields
    util_input_names = encoder_util_input_fields + decoder_util_input_fields

    start_time = time.time()
    exec_time = 0.0
    for batch_id, data in enumerate(train_data()):
        if batch_id >= num_iters:
            break
        feed_list = []
        total_num_token = 0
        for place_id, data_buffer in enumerate(
                split_data(
                    data, num_part=dev_count)):
            data_input_dict, util_input_dict, num_token = prepare_batch_input(
                data_buffer, data_input_names, util_input_names,
                ModelHyperParams.eos_idx, ModelHyperParams.eos_idx,
                ModelHyperParams.n_head, ModelHyperParams.d_model)
            total_num_token += num_token
            feed_kv_pairs = data_input_dict.items() + util_input_dict.items()
            lr_rate = lr_scheduler.update_learning_rate()
            feed_kv_pairs += {lr_scheduler.learning_rate.name: lr_rate}.items()
            feed_list.append(dict(feed_kv_pairs))

            if not init:
                for pos_enc_param_name in pos_enc_param_names:
                    pos_enc = position_encoding_init(
                        ModelHyperParams.max_length + 1,
                        ModelHyperParams.d_model)
                    feed_list[place_id][pos_enc_param_name] = pos_enc
        for feed_dict in feed_list:
            feed_dict[sum_cost.name + "@GRAD"] = 1. / total_num_token

        exe_start_time = time.time()
        if dev_count > 1:
            # prallel executor
            outs = exe.run(fetch_list=[sum_cost.name, token_num.name],
                           feed=feed_list)
        else:
            # executor
            outs = exe.run(fetch_list=[sum_cost, token_num], feed=feed_list[0])
        exec_time += time.time() - exe_start_time

        sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
        total_sum_cost = sum_cost_val.sum()  # sum the cost from multi-devices
        total_token_num = token_num_val.sum()
        total_avg_cost = total_sum_cost / total_token_num
        print("batch: %d, sum loss: %f, avg loss: %f, ppl: %f" %
              (batch_id, total_sum_cost, total_avg_cost,
               np.exp([min(total_avg_cost, 100)])))
        init = True
    return time.time() - start_time, exec_time


def profile(args):
    print args

    if args.device == 'CPU':
        TrainTaskConfig.use_gpu = False

    if not TrainTaskConfig.use_gpu:
        place = fluid.CPUPlace()
        dev_count = multiprocessing.cpu_count()
    else:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()

    exe = fluid.Executor(place)

    sum_cost, avg_cost, predict, token_num = transformer(
        ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
        ModelHyperParams.n_head, ModelHyperParams.d_key,
        ModelHyperParams.d_value, ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid, ModelHyperParams.dropout,
        ModelHyperParams.weight_sharing, TrainTaskConfig.label_smooth_eps)
    lr_scheduler = LearningRateScheduler(ModelHyperParams.d_model,
                                         TrainTaskConfig.warmup_steps,
                                         TrainTaskConfig.learning_rate)

    optimizer = fluid.optimizer.Adam(
        learning_rate=lr_scheduler.learning_rate,
        beta1=TrainTaskConfig.beta1,
        beta2=TrainTaskConfig.beta2,
        epsilon=TrainTaskConfig.eps)
    optimizer.minimize(sum_cost)

    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        fluid.io.load_persistables(exe, TrainTaskConfig.ckpt_path)
        lr_scheduler.current_steps = TrainTaskConfig.start_step
    else:
        exe.run(fluid.framework.default_startup_program())

    # Disable all sorts for they will be done in the 1st batch.
    train_data = reader.DataReader(
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        fpattern=args.train_file_pattern,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size * (1 if args.use_token_batch else dev_count),
        pool_size=args.pool_size,
        sort_type='none',
        shuffle=False,
        shuffle_batch=False,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False)
    train_data = read_multiple(
        reader=train_data.batch_generator,
        count=dev_count if args.use_token_batch else 1)

    if dev_count > 1:
        build_strategy = fluid.BuildStrategy()
        build_strategy.gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
        train_exe = fluid.ParallelExecutor(
            use_cuda=TrainTaskConfig.use_gpu,
            loss_name=sum_cost.name,
            main_program=fluid.default_main_program(),
            build_strategy=build_strategy)

    print("Warming up ...")
    train_loop(exe if dev_count == 1 else train_exe,
               fluid.default_main_program(), False, 3, train_data, dev_count,
               sum_cost, avg_cost, lr_scheduler, token_num, predict)

    print("\nProfiling ...")
    if dev_count == 1:
        with profiler.profiler('All', 'total', '/tmp/profile_file'):
            total_time, exec_time = train_loop(
                exe,
                fluid.default_main_program(), True, args.num_iters, train_data,
                dev_count, sum_cost, avg_cost, lr_scheduler, token_num, predict)
    else:
        total_time, exec_time = train_loop(
            train_exe,
            fluid.default_main_program(), True, args.num_iters, train_data,
            dev_count, sum_cost, avg_cost, lr_scheduler, token_num, predict)
    print("Elapsed time: total %f s, in executor %f s" %
          (total_time, exec_time))


if __name__ == "__main__":
    args = parse_args()
    profile(args)
