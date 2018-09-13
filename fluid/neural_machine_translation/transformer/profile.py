import argparse
import ast
import multiprocessing
import os
import six
import time

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler

import reader
from config import *
from train import pad_batch_data, prepare_data_generator, \
    prepare_feed_dict_list, py_reader_provider_wrapper
from model import transformer, position_encoding_init


def parse_args():
    parser = argparse.ArgumentParser("Training for Transformer.")
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
        "--sort_type",
        default="pool",
        choices=("global", "pool", "none"),
        help="The grain to sort by length: global for all instances; pool for "
        "instances in pool; none for no sort.")
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
        "For EN-DE BPE data we provided, use spaces as token delimiter. "
        "For EN-FR wordpiece data we provided, use '\x01' as token delimiter.")
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
        "--iter_num",
        type=int,
        default=-1,
        help="The iteration number to run in profiling.")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)

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


def main(args):
    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            sum_cost, avg_cost, predict, token_num, pyreader = transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 1,
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
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=False)
            lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
                ModelHyperParams.d_model, TrainTaskConfig.warmup_steps)
            optimizer = fluid.optimizer.Adam(
                learning_rate=lr_decay * TrainTaskConfig.learning_rate,
                beta1=TrainTaskConfig.beta1,
                beta2=TrainTaskConfig.beta2,
                epsilon=TrainTaskConfig.eps)
            optimizer.minimize(avg_cost)

    if args.use_mem_opt:
        fluid.memory_optimize(train_prog)

    if TrainTaskConfig.use_gpu:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)
    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        fluid.io.load_persistables(exe, TrainTaskConfig.ckpt_path)
    else:
        exe.run(startup_prog)

    train_data = prepare_data_generator(
        args, is_test=False, count=dev_count, pyreader=pyreader)

    build_strategy = fluid.BuildStrategy()
    # Since the token number differs among devices, customize gradient scale to
    # use token average cost among multi-devices. and the gradient scale is
    # `1 / token_number` for average cost.
    build_strategy.gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
    train_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        loss_name=avg_cost.name,
        main_program=train_prog,
        build_strategy=build_strategy)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -((1. - TrainTaskConfig.label_smooth_eps) * np.log(
        (1. - TrainTaskConfig.label_smooth_eps
         )) + TrainTaskConfig.label_smooth_eps *
                        np.log(TrainTaskConfig.label_smooth_eps / (
                            ModelHyperParams.trg_vocab_size - 1) + 1e-20))

    def run(iter_num=None, fetch_feq=1):
        reader_time = []
        run_time = []
        step_idx = 0
        init_flag = True
        for pass_id in six.moves.xrange(TrainTaskConfig.pass_num):
            pass_start_time = time.time()

            if args.use_py_reader:
                pyreader.start()
                data_generator = None
            else:
                data_generator = train_data()

            batch_id = 0
            while True:
                try:
                    start_time = time.time()
                    feed_dict_list = prepare_feed_dict_list(
                        data_generator, init_flag, dev_count)
                    end_time = time.time()
                    reader_time.append(end_time - start_time)

                    if step_idx % fetch_feq == fetch_feq - 1:
                        start_time = time.time()
                        outs = train_exe.run(
                            fetch_list=[sum_cost.name, token_num.name],
                            feed=feed_dict_list)
                        end_time = time.time()
                        sum_cost_val, token_num_val = np.array(outs[
                            0]), np.array(outs[1])
                        # sum the cost from multi-devices
                        total_sum_cost = sum_cost_val.sum()
                        total_token_num = token_num_val.sum()
                        total_avg_cost = total_sum_cost / total_token_num

                        print(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                    else:
                        start_time = time.time()
                        outs = train_exe.run(fetch_list=[], feed=feed_dict_list)
                        end_time = time.time()
                    run_time.append(end_time - start_time)

                    init_flag = False
                    batch_id += 1
                    step_idx += 1

                    if iter_num > 0 and step_idx == iter_num:
                        return reader_time, run_time
                except (StopIteration, fluid.core.EOFException):
                    # The current pass is over.
                    if args.use_py_reader:
                        pyreader.reset()
                    break

            pass_end_consumed = time.time()
            print("epoch: %d, consumed %fs" %
                  (pass_id, pass_end_consumed - pass_start_time))

        return reader_time, run_time

    # profiling
    start = time.time()
    # currently only support profiling on one device
    with profiler.profiler('All', 'total', '/tmp/profile_file'):
        reader_time, run_time = run(args.iter_num)
    end = time.time()
    total_time = end - start
    print("Total time: {0}, reader time: {1} s, run time: {2} s".format(
        total_time, np.sum(reader_time), np.sum(run_time)))


if __name__ == "__main__":
    args = parse_args()
    main(args)
