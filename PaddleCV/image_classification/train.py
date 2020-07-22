#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import logging

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import profiler
import reader
from utils import *
import models
from build_model import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_program(is_train, main_prog, startup_prog, args):
    """build program, and add backward op in program accroding to different mode

    Parameters:
        is_train: indicate train mode or test mode
        main_prog: main program
        startup_prog: strartup program
        args: arguments

    Returns : 
        train mode: [Loss, global_lr, data_loader]
        test mode: [Loss, data_loader]
    """
    if args.model.startswith('EfficientNet'):
        override_params = {"drop_connect_rate": args.drop_connect_rate}
        padding_type = args.padding_type
        use_se = args.use_se
        model = models.__dict__[args.model](is_test=not is_train,
                                            override_params=override_params,
                                            padding_type=padding_type,
                                            use_se=use_se)
    else:
        model = models.__dict__[args.model]()
    with fluid.program_guard(main_prog, startup_prog):
        if args.random_seed or args.enable_ce:
            main_prog.random_seed = args.random_seed
            startup_prog.random_seed = args.random_seed
        with fluid.unique_name.guard():
            data_loader, loss_out = create_model(model, args, is_train)
            # add backward op in program
            if is_train:
                optimizer = create_optimizer(args)
                avg_cost = loss_out[0]
                #XXX: fetch learning rate now, better implement is required here. 
                global_lr = optimizer._global_learning_rate()
                global_lr.persistable = True
                loss_out.append(global_lr)

                if args.use_fp16:
                    optimizer = fluid.contrib.mixed_precision.decorate(
                        optimizer,
                        init_loss_scaling=args.scale_loss,
                        use_dynamic_loss_scaling=args.use_dynamic_loss_scaling)

                optimizer.minimize(avg_cost)
                if args.use_ema:
                    global_steps = fluid.layers.learning_rate_scheduler._decay_step_counter(
                    )
                    ema = ExponentialMovingAverage(
                        args.ema_decay, thres_steps=global_steps)
                    ema.update()
                    loss_out.append(ema)
            loss_out.append(data_loader)
    return loss_out


def validate(args,
             test_iter,
             exe,
             test_prog,
             test_fetch_list,
             pass_id,
             train_batch_metrics_record,
             train_batch_time_record=None,
             train_prog=None):
    test_batch_time_record = []
    test_batch_metrics_record = []
    test_batch_id = 0
    if int(os.environ.get('PADDLE_TRAINERS_NUM', 1)) > 1:
        compiled_program = test_prog
    else:
        compiled_program = best_strategy_compiled(
            args,
            test_prog,
            test_fetch_list[0],
            exe,
            mode="val",
            share_prog=train_prog)
    for batch in test_iter:
        t1 = time.time()
        test_batch_metrics = exe.run(program=compiled_program,
                                     feed=batch,
                                     fetch_list=test_fetch_list)
        t2 = time.time()
        test_batch_elapse = t2 - t1
        test_batch_time_record.append(test_batch_elapse)

        test_batch_metrics_avg = np.mean(np.array(test_batch_metrics), axis=1)
        test_batch_metrics_record.append(test_batch_metrics_avg)

        print_info("batch", test_batch_metrics_avg, test_batch_elapse, pass_id,
                   test_batch_id, args.print_step, args.class_dim)
        sys.stdout.flush()
        test_batch_id += 1

    train_epoch_metrics_avg = np.mean(
        np.array(train_batch_metrics_record), axis=0)

    test_epoch_time_avg = np.mean(np.array(test_batch_time_record))
    test_epoch_metrics_avg = np.mean(
        np.array(test_batch_metrics_record), axis=0)

    print_info(
        "epoch",
        list(train_epoch_metrics_avg) + list(test_epoch_metrics_avg),
        test_epoch_time_avg,
        pass_id=pass_id,
        class_dim=args.class_dim)
    if args.enable_ce:
        device_num = fluid.core.get_cuda_device_count() if args.use_gpu else 1
        print_info(
            "ce",
            list(train_epoch_metrics_avg) + list(test_epoch_metrics_avg),
            train_batch_time_record,
            device_num=device_num)


def train(args):
    """Train model
    
    Args:
        args: all arguments.    
    """
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    train_out = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    train_data_loader = train_out[-1]
    if args.use_ema:
        train_fetch_vars = train_out[:-2]
        ema = train_out[-2]
    else:
        train_fetch_vars = train_out[:-1]

    train_fetch_list = [var.name for var in train_fetch_vars]

    if args.validate:
        test_prog = fluid.Program()
        test_out = build_program(
            is_train=False,
            main_prog=test_prog,
            startup_prog=startup_prog,
            args=args)
        test_data_loader = test_out[-1]
        test_fetch_vars = test_out[:-1]

        test_fetch_list = [var.name for var in test_fetch_vars]

        #Create test_prog and set layers' is_test params to True
        test_prog = test_prog.clone(for_test=True)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))

    #init model by checkpoint or pretrianed model.
    init_model(exe, args, train_prog)
    num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    if args.use_dali:
        import dali
        train_iter = dali.train(settings=args)
        if trainer_id == 0:
            test_iter = dali.val(settings=args)
    else:
        imagenet_reader = reader.ImageNetReader(0 if num_trainers > 1 else None)
        train_reader = imagenet_reader.train(settings=args)
        if args.use_gpu:
            if num_trainers <= 1:
                places = fluid.framework.cuda_places()
            else:
                places = place
        else:
            if num_trainers <= 1:
                places = fluid.framework.cpu_places()
            else:
                places = place

        train_data_loader.set_sample_list_generator(train_reader, places)

        if args.validate:
            test_reader = imagenet_reader.val(settings=args)
            test_data_loader.set_sample_list_generator(test_reader, places)

    compiled_train_prog = best_strategy_compiled(args, train_prog,
                                                 train_fetch_vars[0], exe)
    #NOTE: this for benchmark
    total_batch_num = 0
    for pass_id in range(args.num_epochs):
        if num_trainers > 1 and not args.use_dali:
            imagenet_reader.set_shuffle_seed(pass_id + (
                args.random_seed if args.random_seed else 0))
        train_batch_id = 0
        train_batch_time_record = []
        train_batch_metrics_record = []

        if not args.use_dali:
            train_iter = train_data_loader()
            if args.validate:
                test_iter = test_data_loader()

        t1 = time.time()
        for batch in train_iter:
            #NOTE: this is for benchmark
            if args.max_iter and total_batch_num == args.max_iter:
                return
            train_batch_metrics = exe.run(compiled_train_prog,
                                          feed=batch,
                                          fetch_list=train_fetch_list)
            t2 = time.time()
            train_batch_elapse = t2 - t1
            train_batch_time_record.append(train_batch_elapse)

            train_batch_metrics_avg = np.mean(
                np.array(train_batch_metrics), axis=1)
            train_batch_metrics_record.append(train_batch_metrics_avg)
            if trainer_id == 0:
                print_info("batch", train_batch_metrics_avg, train_batch_elapse,
                           pass_id, train_batch_id, args.print_step)
                sys.stdout.flush()
            train_batch_id += 1
            t1 = time.time()
            #NOTE: this for benchmark profiler
            total_batch_num = total_batch_num + 1
            if args.is_profiler and pass_id == 0 and train_batch_id == args.print_step:
                profiler.start_profiler("All")
            elif args.is_profiler and pass_id == 0 and train_batch_id == args.print_step + 5:
                profiler.stop_profiler("total", args.profiler_path)
                return

        if args.use_dali:
            train_iter.reset()

        if trainer_id == 0 and args.validate:
            if args.use_ema:
                logger.info('ExponentialMovingAverage validate start...')
                with ema.apply(exe):
                    validate(args, test_iter, exe, test_prog, test_fetch_list,
                             pass_id, train_batch_metrics_record,
                             compiled_train_prog)
                logger.info('ExponentialMovingAverage validate over!')

            validate(args, test_iter, exe, test_prog, test_fetch_list, pass_id,
                     train_batch_metrics_record, train_batch_time_record,
                     compiled_train_prog)

            if args.use_dali:
                test_iter.reset()

        if trainer_id == 0 and pass_id % args.save_step == 0:
            save_model(args, exe, train_prog, pass_id)


def main():
    args = parse_args()
    if int(os.getenv("PADDLE_TRAINER_ID", 0)) == 0:
        print_arguments(args)
    check_args(args)
    train(args)


if __name__ == '__main__':
    main()
