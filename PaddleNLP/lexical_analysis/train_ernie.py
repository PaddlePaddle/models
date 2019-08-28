"""
Baidu's open-source Lexical Analysis tool for Chinese, including:
    1. Word Segmentation,
    2. Part-of-Speech Tagging
    3. Named Entity Recognition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import multiprocessing
import sys
import warnings

import paddle.fluid as fluid
import nets
import utils


sys.path.append("..")
from models.representation.ernie import ErnieConfig
from models.model_check import check_cuda

def evaluate(exe, test_program, test_pyreader, test_ret):
    """
    Evaluation Function
    """
    test_ret["chunk_evaluator"].reset()
    total_loss = []
    start_time = time.time()
    for data in test_pyreader():

        loss, nums_infer, nums_label, nums_correct = exe.run(
            test_program,
            fetch_list=[
                test_ret["loss"],
                test_ret["num_infer_chunks"],
                test_ret["num_label_chunks"],
                test_ret["num_correct_chunks"],
            ],
            feed=data[0]
        )
        total_loss.append(loss)

        test_ret["chunk_evaluator"].update(nums_infer, nums_label, nums_correct)

    precision, recall, f1 = test_ret["chunk_evaluator"].eval()
    end_time = time.time()
    print("\t[test] loss: %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time: %.3f s"
        % (np.mean(total_loss), precision, recall, f1, end_time - start_time))


def do_train(args):
    """
    Main Function
    """
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        dev_count = min(multiprocessing.cpu_count(), args.cpu_num)
        if (dev_count < args.cpu_num):
            warnings.warn(
                'The total CPU NUM in this machine is %d, which is less than cpu_num parameter you set' % dev_count)
            warnings.warn('Change the cpu_num from %d to %d' % (args.cpu_num, dev_count))
        os.environ['CPU_NUM'] = str(dev_count)
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)


    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    # num_train_examples = reader.get_num_examples(args.train_set)
    # max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count
    print("Device count: %d" % dev_count)
    # print("Num train examples: %d" % num_train_examples)
    # print("Max train steps: %d" % max_train_steps)

    train_program = fluid.Program()

    with fluid.program_guard(train_program, startup_prog):
        with fluid.unique_name.guard():
            # user defined model based on ernie embeddings
            train_ret = nets.create_ernie_model(args, ernie_config, is_prediction=False)

            # ernie pyreader
            train_pyreader = nets.create_pyreader(args, file_name=args.train_data,
                                                  feed_list=train_ret['feed_list'],
                                                  mode="ernie",
                                                  place=place,
                                                  iterable=True)

            test_program = train_program.clone(for_test=True)
            test_pyreader = nets.create_pyreader(args, file_name=args.test_data,
                                                  feed_list=train_ret['feed_list'],
                                                  mode="ernie",
                                                  place=place,
                                                  iterable=True)

            optimizer = fluid.optimizer.Adam(learning_rate=args.base_learning_rate)
            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0))
            optimizer.minimize(train_ret["loss"])

    lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
        program=train_program, batch_size=args.batch_size)
    print("Theoretical memory usage in training: %.3f - %.3f %s" %
        (lower_mem, upper_mem, unit))


    exe.run(startup_prog)

    # load checkpoints
    if args.init_checkpoint and args.init_pretraining_params:
        print("WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
    if args.init_checkpoint:
        utils.init_checkpoint(exe, args.init_checkpoint, startup_prog)
    elif args.init_pretraining_params:
        utils.init_pretraining_params(exe, args.init_pretraining_params, startup_prog)

    if dev_count>1 and not args.use_cuda:
        device = "GPU" if args.use_cuda else "CPU"
        print("%d %s are used to train model"%(dev_count, device))
        # multi cpu/gpu config
        exec_strategy = fluid.ExecutionStrategy()
        # exec_strategy.num_threads = dev_count * 6
        build_strategy = fluid.BuildStrategy()
        # build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        # build_strategy.fuse_broadcast_ops = True
        # build_strategy.fuse_elewise_add_act_ops = True
        # build_strategy.enable_inplace = True

        compiled_prog = fluid.compiler.CompiledProgram(train_program).with_data_parallel(
            loss_name=train_ret['loss'].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy
        )
    else:
        compiled_prog = fluid.compiler.CompiledProgram(train_program)


    steps = 0
    for epoch_id in range(args.epoch):
        for data in train_pyreader():
            steps += 1
            if steps % args.print_steps == 0:
                fetch_list = [
                    train_ret["loss"],
                    train_ret["precision"],
                    train_ret["recall"],
                    train_ret["f1_score"],
                ]
            else:
                fetch_list = []

            start_time = time.time()
            outputs = exe.run(program=compiled_prog, feed=data[0], fetch_list=fetch_list)
            end_time = time.time()
            if steps % args.print_steps == 0:
                loss, precision, recall, f1_score = [np.mean(x) for x in outputs]
                print("[train] batch_id = %d, loss = %.5f, P: %.5f, R: %.5f, F1: %.5f, elapsed time %.5f, "
                        "pyreader queue_size: %d " % (steps, loss, precision, recall, f1_score,
                        end_time - start_time, train_pyreader.queue.size()))

            if steps % args.save_steps == 0:
                save_path = os.path.join(args.model_save_dir, "step_" + str(steps))
                print("\tsaving model as %s" % (save_path))
                fluid.io.save_persistables(exe, save_path, train_program)

            if steps % args.validation_steps == 0:
                # evaluate test set
                if args.do_test:
                    evaluate(exe, test_program, test_pyreader, train_ret)


    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
    fluid.io.save_persistables(exe, save_path, train_program)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    utils.load_yaml(parser, 'ernie_args.yaml')
    args = parser.parse_args()
    check_cuda(args.use_cuda)
    utils.print_arguments(args)
    do_train(args)
