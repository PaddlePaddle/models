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

import logging
import os
import six
import sys
import time

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import profiler

import utils.dist_utils as dist_utils
from utils.input_field import InputField
from utils.configure import PDConfig
from utils.check import check_gpu, check_version

# include task-specific libs
import desc
import reader
from transformer import create_net, position_encoding_init

if os.environ.get('FLAGS_eager_delete_tensor_gb', None) is None:
    os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
# num_trainers is used for multi-process gpu training
num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))


def init_from_pretrain_model(args, exe, program):

    assert isinstance(args.init_from_pretrain_model, str)

    if not os.path.exists(args.init_from_pretrain_model):
        raise Warning("The pretrained params do not exist.")
        return False

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(
            os.path.join(args.init_from_pretrain_model, var.name))

    fluid.io.load_vars(
        exe,
        args.init_from_pretrain_model,
        main_program=program,
        predicate=existed_params)

    print("finish initing model from pretrained params from %s" %
          (args.init_from_pretrain_model))

    return True


def init_from_checkpoint(args, exe, program):

    assert isinstance(args.init_from_checkpoint, str)

    if not os.path.exists(args.init_from_checkpoint):
        raise Warning("the checkpoint path does not exist.")
        return False

    fluid.io.load_persistables(
        executor=exe,
        dirname=args.init_from_checkpoint,
        main_program=program,
        filename="checkpoint.pdckpt")

    print("finish initing model from checkpoint from %s" %
          (args.init_from_checkpoint))

    return True


def save_checkpoint(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)

    checkpoint_dir = os.path.join(args.save_model_path, args.save_checkpoint)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    fluid.io.save_persistables(
        exe,
        os.path.join(checkpoint_dir, dirname),
        main_program=program,
        filename="checkpoint.pdparams")

    print("save checkpoint at %s" % (os.path.join(checkpoint_dir, dirname)))

    return True


def save_param(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)

    param_dir = os.path.join(args.save_model_path, args.save_param)

    if not os.path.exists(param_dir):
        os.mkdir(param_dir)

    fluid.io.save_params(
        exe,
        os.path.join(param_dir, dirname),
        main_program=program,
        filename="params.pdparams")
    print("save parameters at %s" % (os.path.join(param_dir, dirname)))

    return True


def do_train(args):
    if args.use_cuda:
        if num_trainers > 1:  # for multi-process gpu training
            dev_count = 1
        else:
            dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = fluid.CUDAPlace(gpu_id)
    else:
        dev_count = int(os.environ.get('CPU_NUM', 1))
        place = fluid.CPUPlace()

    # define the data generator
    processor = reader.DataProcessor(
        fpattern=args.training_file,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size,
        device_count=dev_count,
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        max_length=args.max_length,
        n_head=args.n_head)
    batch_generator = processor.data_generator(phase="train")
    if num_trainers > 1:  # for multi-process gpu training
        batch_generator = fluid.contrib.reader.distributed_batch_reader(
            batch_generator)
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()

    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        train_prog.random_seed = random_seed
        startup_prog.random_seed = random_seed

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():

            # define input and reader

            input_field_names = desc.encoder_data_input_fields + \
                    desc.decoder_data_input_fields[:-1] + desc.label_data_input_fields
            input_descs = desc.get_input_descs(args.args)
            input_slots = [{
                "name": name,
                "shape": input_descs[name][0],
                "dtype": input_descs[name][1]
            } for name in input_field_names]

            input_field = InputField(input_slots)
            input_field.build(build_pyreader=True)

            # define the network

            sum_cost, avg_cost, token_num = create_net(
                is_training=True, model_input=input_field, args=args)

            # define the optimizer

            with fluid.default_main_program()._lr_schedule_guard():
                learning_rate = fluid.layers.learning_rate_scheduler.noam_decay(
                    args.d_model, args.warmup_steps) * args.learning_rate

            optimizer = fluid.optimizer.Adam(
                learning_rate=learning_rate,
                beta1=args.beta1,
                beta2=args.beta2,
                epsilon=float(args.eps))
            optimizer.minimize(avg_cost)

    # prepare training

    ## decorate the pyreader with batch_generator
    input_field.loader.set_batch_generator(batch_generator)

    ## define the executor and program for training

    exe = fluid.Executor(place)

    exe.run(startup_prog)
    # init position_encoding
    for pos_enc_param_name in desc.pos_enc_param_names:
        pos_enc_param = fluid.global_scope().find_var(
            pos_enc_param_name).get_tensor()

        pos_enc_param.set(
            position_encoding_init(args.max_length + 1, args.d_model), place)

    assert (args.init_from_checkpoint == "") or (
        args.init_from_pretrain_model == "")

    ## init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        init_from_checkpoint(args, exe, train_prog)

    ## init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        init_from_pretrain_model(args, exe, train_prog)

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.enable_inplace = True
    exec_strategy = fluid.ExecutionStrategy()
    if num_trainers > 1:
        dist_utils.prepare_for_multi_process(exe, build_strategy, train_prog)
        exec_strategy.num_threads = 1

    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=avg_cost.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1. - args.label_smooth_eps) * np.log(
            (1. - args.label_smooth_eps)) + args.label_smooth_eps *
        np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))
    # start training

    step_idx = 0
    total_batch_num = 0  # this is for benchmark
    for pass_id in range(args.epoch):
        pass_start_time = time.time()
        input_field.loader.start()

        batch_id = 0
        while True:
            if args.max_iter and total_batch_num == args.max_iter: # this for benchmark
                return
            try:
                outs = exe.run(compiled_train_prog,
                               fetch_list=[sum_cost.name, token_num.name]
                               if step_idx % args.print_step == 0 else [])

                if step_idx % args.print_step == 0:
                    sum_cost_val, token_num_val = np.array(outs[0]), np.array(
                        outs[1])
                    # sum the cost from multi-devices
                    total_sum_cost = sum_cost_val.sum()
                    total_token_num = token_num_val.sum()
                    total_avg_cost = total_sum_cost / total_token_num

                    if step_idx == 0:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                        avg_batch_time = time.time()
                    else:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, speed: %.2f step/s" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)]),
                             args.print_step / (time.time() - avg_batch_time)))
                        avg_batch_time = time.time()

                if step_idx % args.save_step == 0 and step_idx != 0:

                    if args.save_checkpoint:
                        save_checkpoint(args, exe, train_prog,
                                        "step_" + str(step_idx))

                    if args.save_param:
                        save_param(args, exe, train_prog,
                                   "step_" + str(step_idx))

                batch_id += 1
                step_idx += 1
                total_batch_num = total_batch_num + 1 # this is for benchmark

                # profiler tools for benchmark
                if args.is_profiler and pass_id == 0 and batch_id == args.print_step:
                    profiler.start_profiler("All")
                elif args.is_profiler and pass_id == 0 and batch_id == args.print_step + 5:
                    profiler.stop_profiler("total", args.profiler_path)
                    return

            except fluid.core.EOFException:
                input_field.loader.reset()
                break

        time_consumed = time.time() - pass_start_time

    if args.save_checkpoint:
        save_checkpoint(args, exe, train_prog, "step_final")

    if args.save_param:
        save_param(args, exe, train_prog, "step_final")

    if args.enable_ce:  # For CE
        print("kpis\ttrain_cost_card%d\t%f" % (dev_count, total_avg_cost))
        print("kpis\ttrain_duration_card%d\t%f" % (dev_count, time_consumed))


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_train(args)
