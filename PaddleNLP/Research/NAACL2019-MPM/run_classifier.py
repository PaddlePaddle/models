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
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import argparse
import numpy as np
import multiprocessing

import paddle
import paddle.fluid as fluid
from classifier import create_model
import reader

sys.path.append("./BERT")
from model.bert import BertConfig
from optimization import optimization
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params, init_checkpoint
import scipy
from sklearn.model_selection import KFold, StratifiedKFold

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("bert_config_path", str, None, "Path to the json file for bert model config.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_proportion", float, 0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
train_g.add_arg("loss_scaling", float, 1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir", str, None, "Path to training data.")
data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest seqence.")
data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens", bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("random_seed", int, 0, "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int, 1, "Ihe iteration intervals to clean up temporary variables.")
run_type_g.add_arg("task_name", str, "sem",
                   "The name of task to perform fine-tuning, should be in {'xnli', 'mnli', 'cola', 'mrpc'}.")
run_type_g.add_arg("sub_model_type", str, "raw",
                   "The type of sub model to use, should be in {'raw', 'cnn', 'gru', ffa}.")
run_type_g.add_arg("ksplit", int, -1,
                   "if ksplit > 0, use kfold training")
run_type_g.add_arg("drop_keyword", bool, False,
                   "if drop keyword for data augmentation.")
run_type_g.add_arg("kfold_type", str, "normal",
                   "The type of kfold should be in {'normal', 'stratified'}")
run_type_g.add_arg("binary", bool, True, "if is binary classification.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation on dev data set.")
run_type_g.add_arg("do_test", bool, True, "Whether to perform evaluation on test data set.")

args = parser.parse_args()


# yapf: enable.


def evaluate(exe, test_program, test_pyreader, fetch_list, eval_phase):
    """
    evaluation for dev and test dataset.
    """
    test_pyreader.start()
    total_cost, total_acc, total_num_seqs = 0.0, 0.0, 0.0
    total_label_pos_num, total_pred_pos_num, total_correct_num = 0.0, 0.0, 0.0
    qids, labels, scores = [], [], []
    time_begin = time.time()
    while True:
        try:
            np_loss, np_acc, np_probs, np_labels, np_num_seqs = exe.run(
                program=test_program, fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_acc += np.sum(np_acc * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            scores.extend(np_probs[:, 1].reshape(-1).tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            total_label_pos_num += np.sum(np_labels)
            total_pred_pos_num += np.sum(np_preds)
            total_correct_num += np.sum(np.dot(np_preds, np_labels))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    r = total_correct_num / total_label_pos_num
    p = total_correct_num / total_pred_pos_num
    f = 2 * p * r / (p + r)

    print(
        "[%s evaluation] ave loss: %f, ave_acc: %f, p: %f, r: %f, f1: %f, data_num: %d, elapsed time: %f s"
        % (eval_phase, total_cost / total_num_seqs,
           total_acc / total_num_seqs, p, r, f, total_num_seqs,
           time_end - time_begin))


def predict(exe, test_program, test_pyreader, fetch_list, eval_phase, output_file):
    """
    predict function
    """
    test_pyreader.start()
    qids, scores = [], []
    time_begin = time.time()
    while True:
        try:
            np_probs, np_num_seqs = exe.run(
                program=test_program, fetch_list=fetch_list)
            scores.extend(np_probs[:, 1].reshape(-1).tolist())
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    with open(output_file, 'w') as w:
        for prob in scores:
            w.write(str(prob) + '\n')


def train_kfold(args):
    """
    main program for training kfold.
    """
    task_name = args.task_name.lower()
    processors = {
        'sem': reader.SemevalTask9Processor,
    }

    processor = processors[task_name](data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case,
                                      in_tokens=args.in_tokens,
                                      random_seed=args.random_seed)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    train_examples = processor.get_train_examples(args.data_dir, drop_keyword=args.drop_keyword)
    test_examples = processor.get_test_examples(args.data_dir)

    if args.kfold_type == 'normal':
        kf = KFold(n_splits=args.ksplit, shuffle=True, random_state=args.random_seed)
        kf_iter = kf.split(train_examples)
    elif args.kfold_type == 'stratified':
        kf = StratifiedKFold(n_splits=args.ksplit, shuffle=True, random_state=args.random_seed)
        train_labels = [e.label for e in train_examples]
        kf_iter = kf.split(train_examples, train_labels)
    else:
        raise NotImplementedError("%s is not implemented" % args.kfold_type)

    for fold, (train_idx, val_idx) in enumerate(kf_iter):
        print("==================== fold %d ===================" % fold)
        train_fold = np.array(train_examples)[train_idx]
        dev_fold = np.array(train_examples)[val_idx]
        test_examples = np.array(test_examples)
        kfold_program(args, processor, train_fold, dev_fold, test_examples, str(fold))


def kfold_program(args, processor, train_examples, dev_examples, test_examples, fold):
    """
    training program for kfold.
    """
    bert_config = BertConfig(args.bert_config_path)
    bert_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    num_labels = len(processor.get_labels())

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = processor.data_generator_for_kfold(
            examples=train_examples,
            batch_size=args.batch_size,
            phase='train',
            epoch=args.epoch,
            dev_count=dev_count,
            shuffle=True)

        num_train_examples = processor.get_num_examples(phase='train')

        if args.in_tokens:
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
                train_pyreader, loss, probs, accuracy, labels, num_seqs = create_model(
                    args,
                    pyreader_name=fold + 'train_reader',
                    bert_config=bert_config,
                    num_labels=num_labels)
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

        if args.verbose:
            if args.in_tokens:
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
                test_pyreader, loss, probs, accuracy, labels, num_seqs = create_model(
                    args,
                    pyreader_name=fold + 'test_reader',
                    bert_config=bert_config,
                    num_labels=num_labels)

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
                        fetch_list = [loss.name, accuracy.name, num_seqs.name]
                    else:
                        fetch_list = [
                            loss.name, accuracy.name, scheduled_lr.name,
                            num_seqs.name
                        ]
                else:
                    fetch_list = []

                outputs = train_exe.run(fetch_list=fetch_list)

                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        np_loss, np_acc, np_num_seqs = outputs
                    else:
                        np_loss, np_acc, np_lr, np_num_seqs = outputs

                    total_cost.extend(np_loss * np_num_seqs)
                    total_acc.extend(np_acc * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            np_lr[0]
                            if warmup_steps > 0 else args.learning_rate)
                        print(verbose)

                    current_example, current_epoch = processor.get_train_progress(
                    )
                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                          "ave acc: %f, speed: %f steps/s" %
                          (current_epoch, current_example, num_train_examples,
                           steps, np.sum(total_cost) / np.sum(total_num_seqs),
                           np.sum(total_acc) / np.sum(total_num_seqs),
                           args.skip_steps / used_time))
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints, fold,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        test_pyreader.decorate_tensor_provider(
                            processor.data_generator_for_kfold(
                                examples=dev_examples,
                                batch_size=args.batch_size,
                                phase='dev',
                                epoch=1,
                                dev_count=1,
                                shuffle=False))
                        evaluate(exe, test_prog, test_pyreader,
                                 [loss.name, accuracy.name, probs.name, labels.name, num_seqs.name],
                                 "dev")
                    # evaluate test set
                    if args.do_test:
                        test_pyreader.decorate_tensor_provider(
                            processor.data_generator_for_kfold(
                                examples=test_examples,
                                batch_size=args.batch_size,
                                phase='test',
                                epoch=1,
                                dev_count=1,
                                shuffle=False))
                        evaluate(exe, test_prog, test_pyreader,
                                 [loss.name, accuracy.name, probs.name, labels.name, num_seqs.name],
                                 "test")
            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, fold, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        test_pyreader.decorate_tensor_provider(
            processor.data_generator_for_kfold(
                examples=dev_examples,
                batch_size=args.batch_size, phase='dev', epoch=1, dev_count=1,
                shuffle=False))
        print("Final validation result:")
        evaluate(exe, test_prog, test_pyreader,
                 [loss.name, accuracy.name, probs.name, labels.name, num_seqs.name], "dev")

    # final eval on test set
    if args.do_test:
        test_pyreader.decorate_tensor_provider(
            processor.data_generator_for_kfold(
                examples=test_examples,
                batch_size=args.batch_size,
                phase='test',
                epoch=1,
                dev_count=1,
                shuffle=False))
        print("Final test result:")
        evaluate(exe, test_prog, test_pyreader,
                 [loss.name, accuracy.name, probs.name, labels.name, num_seqs.name], "test")

    exe.close()


def train_single(args):
    """
    training program.
    """
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
    processors = {
        'sem': reader.SemevalTask9Processor,
    }

    processor = processors[task_name](data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      max_seq_len=args.max_seq_len,
                                      do_lower_case=args.do_lower_case,
                                      in_tokens=args.in_tokens,
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
            dev_count=dev_count,
            shuffle=True,
            drop_keyword=args.drop_keyword)

        num_train_examples = processor.get_num_examples(phase='train')

        if args.in_tokens:
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
                train_pyreader, loss, probs, accuracy, labels, num_seqs = create_model(
                    args,
                    pyreader_name='train_reader',
                    bert_config=bert_config,
                    num_labels=num_labels)
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

        if args.verbose:
            if args.in_tokens:
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
                test_pyreader, loss, probs, accuracy, labels, num_seqs = create_model(
                    args,
                    pyreader_name='test_reader',
                    bert_config=bert_config,
                    num_labels=num_labels)

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
                        fetch_list = [loss.name, accuracy.name, num_seqs.name]
                    else:
                        fetch_list = [
                            loss.name, accuracy.name, scheduled_lr.name,
                            num_seqs.name
                        ]
                else:
                    fetch_list = []

                outputs = train_exe.run(fetch_list=fetch_list)

                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        np_loss, np_acc, np_num_seqs = outputs
                    else:
                        np_loss, np_acc, np_lr, np_num_seqs = outputs

                    total_cost.extend(np_loss * np_num_seqs)
                    total_acc.extend(np_acc * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            np_lr[0]
                            if warmup_steps > 0 else args.learning_rate)
                        print(verbose)

                    current_example, current_epoch = processor.get_train_progress(
                    )
                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("epoch: %d, progress: %d/%d, step: %d, ave loss: %f, "
                          "ave acc: %f, speed: %f steps/s" %
                          (current_epoch, current_example, num_train_examples,
                           steps, np.sum(total_cost) / np.sum(total_num_seqs),
                           np.sum(total_acc) / np.sum(total_num_seqs),
                           args.skip_steps / used_time))
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        test_pyreader.decorate_tensor_provider(
                            processor.data_generator(
                                batch_size=args.batch_size,
                                phase='dev',
                                epoch=1,
                                dev_count=1,
                                shuffle=False))
                        evaluate(exe, test_prog, test_pyreader,
                                 [loss.name, accuracy.name, probs.name, labels.name, num_seqs.name],
                                 "dev")
                    # evaluate test set
                    if args.do_test:
                        test_pyreader.decorate_tensor_provider(
                            processor.data_generator(
                                batch_size=args.batch_size,
                                phase='test',
                                epoch=1,
                                dev_count=1,
                                shuffle=False))
                        evaluate(exe, test_prog, test_pyreader,
                                 [loss.name, accuracy.name, probs.name, labels.name, num_seqs.name],
                                 "test")
            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        test_pyreader.decorate_tensor_provider(
            processor.data_generator(
                batch_size=args.batch_size, phase='dev', epoch=1, dev_count=1,
                shuffle=False))
        print("Final validation result:")
        evaluate(exe, test_prog, test_pyreader,
                 [loss.name, accuracy.name, probs.name, labels.name, num_seqs.name], "dev")

    # final eval on test set
    if args.do_test:
        test_pyreader.decorate_tensor_provider(
            processor.data_generator(
                batch_size=args.batch_size,
                phase='test',
                epoch=1,
                dev_count=1,
                shuffle=False))
        print("Final test result:")
        predict(exe, test_prog, test_pyreader,
                [probs.name, num_seqs.name], "test", args.checkpoints + '/prob.txt')


if __name__ == '__main__':
    print_arguments(args)
    if args.ksplit <= 0:
        train_single(args)
    else:
        train_kfold(args)
