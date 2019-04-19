#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""Finetuning on dialogue tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid

from finetune_args import parser
import reader.data_reader as reader
from optimization import optimization
from utils.args import print_arguments
from utils.init import init_checkpoint, init_pretraining_params

_WORK_DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append('../../models/dialogue_model_toolkit/dialogue_general_understanding')

from bert import BertConfig, BertModel
from create_model import create_model
import define_paradigm


def evaluate(test_exe, test_program, test_pyreader, fetch_list, eval_phase):
    """evaluate validation or test data"""
    test_pyreader.start()
    total_cost, total_acc, total_num_seqs = [], [], []
    time_begin = time.time()
    while True:
        try: 
            if len(fetch_list) > 2: 
                np_loss, np_acc, np_num_seqs = test_exe.run(fetch_list=fetch_list)
                total_acc.extend(np_acc * np_num_seqs)
            else: 
                np_loss, np_num_seqs = test_exe.run(fetch_list=fetch_list)
            total_cost.extend(np_loss * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    if len(fetch_list) > 2: 
        print("[%s evaluation] %s ave loss: %f, ave acc: %f, elapsed time: %f s" %
              (eval_phase, current_time, np.sum(total_cost) / np.sum(total_num_seqs),
               np.sum(total_acc) / np.sum(total_num_seqs), time_end - time_begin))
    else: 
        print("[%s evaluation] %s ave loss: %f, elapsed time: %f s" %
             (eval_phase, current_time, np.sum(total_cost) / np.sum(total_num_seqs),
             time_end - time_begin))


def main(args): 
    """main function"""
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    if args.use_cuda: 
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else: 
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    task_name = args.task_name.lower()
    paradigm_inst = define_paradigm.Paradigm(task_name)

    processors = {
        'udc': reader.UDCProcessor,
        'swda': reader.SWDAProcessor,
        'mrda': reader.MRDAProcessor,
        'atis_slot': reader.ATISSlotProcessor,
        'atis_intent': reader.ATISIntentProcessor,
        'dstc2': reader.DSTC2Processor,
    }
    in_tokens = {
        'udc': True,
        'swda': True,
        'mrda': True,
        'atis_slot': False,
        'atis_intent': True,
        'dstc2': True,
    }

    processor = processors[task_name](data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case, 
                                      in_tokens=in_tokens[task_name],
                                      task_name=task_name, 
                                      random_seed=args.random_seed)

    num_labels = len(processor.get_labels())

    if not (args.do_train or args.do_val or args.do_test): 
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train: 
        train_data_generator = processor.data_generator(
            batch_size=args.batch_size,
            phase='train',
            epoch=args.epoch,
            shuffle=True)
        num_train_examples = processor.get_num_examples(phase='train')

        if in_tokens[task_name]: 
            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // dev_count
        else: 
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                results = create_model(
                    args,
                    pyreader_name='train_reader',
                    bert_config=bert_config,
                    num_labels=num_labels,
                    paradigm_inst=paradigm_inst)
                train_pyreader = results.get("pyreader", None)
                loss = results.get("loss", None)
                probs = results.get("probs", None)
                accuracy = results.get("accuracy", None)
                num_seqs = results.get("num_seqs", None)
                scheduled_lr = optimization(
                    loss=loss,
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    loss_scaling=args.loss_scaling)

                if accuracy is not None: 
                    skip_opt_set = [loss.name, probs.name, accuracy.name, num_seqs.name]
                else: 
                    skip_opt_set = [loss.name, probs.name, num_seqs.name]
                fluid.memory_optimize(
                    input_program=train_program, 
                    skip_opt_set=skip_opt_set)

        if args.verbose: 
            if in_tokens[task_name]: 
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else: 
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                (lower_mem, upper_mem, unit))

    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_results = create_model(
                    args,
                    pyreader_name='test_reader',
                    bert_config=bert_config,
                    num_labels=num_labels,
                    paradigm_inst=paradigm_inst)
                test_pyreader = test_results.get("pyreader", None)
                loss = test_results.get("loss", None)
                probs = test_results.get("probs", None)
                accuracy = test_results.get("accuracy", None)
                num_seqs = test_results.get("num_seqs", None)
        test_prog = test_prog.clone(for_test=True)
    
    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            print(
                  "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                  "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(
                exe, 
                args.init_checkpoint, 
                main_program=startup_prog,
                use_fp16=args.use_fp16)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe, 
                args.init_pretraining_params, 
                main_program=startup_prog,
                use_fp16=args.use_fp16)
    elif args.do_val or args.do_test: 
        if not args.init_checkpoint: 
            raise ValueError("args 'init_checkpoint' should be set if"
                    "only doing validation or testing!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog,
            use_fp16=args.use_fp16)

    if args.do_train: 
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            main_program=train_program)
        train_pyreader.decorate_tensor_provider(train_data_generator)
    else: 
        train_exe = None

    if args.do_val or args.do_test:  
        test_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            main_program=test_prog,
            share_vars_from=train_exe)
   
    if args.do_train: 
        train_pyreader.start()
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        time_begin = time.time()
        while True:
            try: 
                steps += 1
                if steps % args.skip_steps == 0: 
                    if warmup_steps <= 0: 
                        if accuracy is not None: 
                            fetch_list = [loss.name, accuracy.name, num_seqs.name]
                        else: 
                            fetch_list = [loss.name, num_seqs.name]
                    else: 
                        if accuracy is not None:
                            fetch_list = [
                                loss.name, accuracy.name, scheduled_lr.name,
                                num_seqs.name
                            ]
                        else: 
                            fetch_list = [loss.name, scheduled_lr.name, num_seqs.name]
                else: 
                    fetch_list = []
                if accuracy is not None: 
                    fetch_test_list = [loss.name, accuracy.name, num_seqs.name]
                else: 
                    fetch_test_list = [loss.name, num_seqs.name]

                outputs = train_exe.run(fetch_list=fetch_list)

                if steps % args.skip_steps == 0: 
                    if warmup_steps <= 0: 
                        if accuracy is not None: 
                            np_loss, np_acc, np_num_seqs = outputs
                        else: 
                            np_loss, np_num_seqs = outputs
                    else: 
                        if accuracy is not None:
                            np_loss, np_acc, np_lr, np_num_seqs = outputs
                        else: 
                            np_loss, np_lr, np_num_seqs = outputs

                    total_cost.extend(np_loss * np_num_seqs) 
                    total_num_seqs.extend(np_num_seqs)
                    if accuracy is not None: 
                        total_acc.extend(np_acc * np_num_seqs)
                    
                    if args.verbose: 
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size()
                        verbose += "learning rate: %f" % (
                            np_lr[0]
                            if warmup_steps > 0 else args.learning_rate)
                        print(verbose) 

                    current_example, current_epoch = processor.get_train_progress()
                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    if accuracy is not None: 
                        print("%s epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                              "ave acc: %f, speed: %f steps/s" %
                              (current_time, current_epoch, current_example, num_train_examples,
                               steps, np.sum(total_cost) / np.sum(total_num_seqs),
                               np.sum(total_acc) / np.sum(total_num_seqs),
                               args.skip_steps / used_time))
                    else: 
                        print("%s epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                            "speed: %f steps/s" %
                            (current_time, current_epoch, current_example, num_train_examples,
                            steps, np.sum(total_cost) / np.sum(total_num_seqs),
                            args.skip_steps / used_time))
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
                if steps % args.validation_steps == 0: 
                    #evaluate dev set
                    if args.do_val:
                        test_pyreader.decorate_tensor_provider(
                            processor.data_generator(  
                                batch_size=args.batch_size,
                                phase='dev',
                                epoch=1,
                                shuffle=False))
                        evaluate(test_exe, test_prog, test_pyreader, fetch_test_list, "dev")
                    #evaluate test set
                    if args.do_test: 
                        test_pyreader.decorate_tensor_provider(
                            processor.data_generator(
                                batch_size=args.batch_size,
                                phase='test',
                                epoch=1,
                                shuffle=False))
                        evaluate(test_exe, test_prog, test_pyreader, fetch_test_list, "test")
            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break
    #final eval on dev set
    if args.do_val: 
        test_pyreader.decorate_tensor_provider( 
            processor.data_generator( 
                batch_size=args.batch_size, phase='dev', epoch=1,
                shuffle=False))
        print("Final validation result:")
        evaluate(test_exe, test_prog, test_pyreader, fetch_test_list, "dev")

    #final eval on test set
    if args.do_test: 
        test_pyreader.decorate_tensor_provider( 
            processor.data_generator(
                batch_size=args.batch_size,
                phase='test',
                epoch=1,
                shuffle=False)) 
        print("Final test result:") 
        evaluate(test_exe, test_prog, test_pyreader, fetch_test_list, "test")


if __name__ == '__main__': 
    args = parser.parse_args()
    print_arguments(args)
    main(args)
