"""
Sentiment Classification Task
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
sys.path.append("../models/classification/")

from nets import bow_net
from nets import lstm_net
from nets import cnn_net
from nets import bilstm_net
from nets import gru_net

import paddle
import paddle.fluid as fluid

import reader
from config import SentaConfig
from utils import ArgumentGroup, print_arguments
from utils import init_checkpoint

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("senta_config_path", str, None, "Path to the json file for senta model config.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
train_g.add_arg("lr", float, 0.002, "The Learning rate value for training.")

log_g = ArgumentGroup(parser, "logging", "logging related")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir", str, None, "Path to training data.")
data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
data_g.add_arg("batch_size", int, 256, "Total examples' number in batch for training.")
data_g.add_arg("random_seed", int, 0, "Random seed.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("task_name", str, None,
    "The name of task to perform sentiment classification.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation.")
run_type_g.add_arg("do_infer", bool, True, "Whether to perform inference.")

args = parser.parse_args()
# yapf: enable.

def create_model(args,
                 pyreader_name,
                 senta_config,
                 num_labels,
                 is_inference=False):

    """
    Create Model for sentiment classification
    """

    pyreader = fluid.layers.py_reader(
        capacity=16,
        shapes=([-1, 1], [-1, 1]),
        dtypes=('int64', 'int64'),
        lod_levels=(1, 0),
        name=pyreader_name,
        use_double_buffer=False)

    if senta_config['model_type'] == "bilstm_net":
        network = bilstm_net
    elif senta_config['model_type'] == "bow_net":
        network = bow_net
    elif senta_config['model_type'] == "cnn_net":
        network = cnn_net
    elif senta_config['model_type'] == "lstm_net":
        network = lstm_net
    elif senta_config['model_type'] == "gru_net":
        network = gru_net
    else:
        raise ValueError("Unknown network type!")

    if is_inference:
        data, label = fluid.layers.read_file(pyreader)
        probs = network(data, None, senta_config["vocab_size"], is_infer=is_inference)
        print("create inference model...")
        return pyreader, probs

    data, label = fluid.layers.read_file(pyreader)
    ce_loss, probs = network(data, label, senta_config["vocab_size"], is_infer=is_inference)
    loss = fluid.layers.mean(x=ce_loss)
    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=label, total=num_seqs)
    return pyreader, loss, accuracy, num_seqs



def evaluate(exe, test_program, test_pyreader, fetch_list, eval_phase):
    """
    Evaluation Function
    """
    test_pyreader.start()
    total_cost, total_acc, total_num_seqs = [], [], []
    time_begin = time.time()
    while True:
        try:
            np_loss, np_acc, np_num_seqs = exe.run(program=test_program,
                                                   fetch_list=fetch_list,
                                                   return_numpy=False)
            np_loss = np.array(np_loss)
            np_acc = np.array(np_acc)
            np_num_seqs = np.array(np_num_seqs)
            total_cost.extend(np_loss * np_num_seqs)
            total_acc.extend(np_acc * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    print("[%s evaluation] ave loss: %f, ave acc: %f, elapsed time: %f s" %
        (eval_phase, np.sum(total_cost) / np.sum(total_num_seqs),
        np.sum(total_acc) / np.sum(total_num_seqs), time_end - time_begin))


def inference(exe, test_program, test_pyreader, fetch_list, infer_phrase):
    """
    Inference Function
    """
    test_pyreader.start()
    time_begin = time.time()
    while True:
        try:
            np_props = exe.run(program=test_program, fetch_list=fetch_list,
                                return_numpy=True)
            for probs in np_props[0]:
                print("%d\t%f\t%f" % (np.argmax(probs), probs[0], probs[1]))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    print("[%s] elapsed time: %f s" % (infer_phrase, time_end - time_begin))


def main(args):
    """
    Main Function
    """
    senta_config = SentaConfig(args.senta_config_path)

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = 1
    exe = fluid.Executor(place)

    task_name = args.task_name.lower()
    processor = reader.SentaProcessor(data_dir=args.data_dir,
                                      vocab_path=args.vocab_path,
                                      random_seed=args.random_seed)
    num_labels = len(processor.get_labels())


    if not (args.do_train or args.do_val or args.do_infer):
        raise ValueError("For args `do_train`, `do_val` and `do_infer`, at "
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

        num_train_examples = processor.get_num_examples(phase="train")

        max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, loss, accuracy, num_seqs = create_model(
                    args,
                    pyreader_name='train_reader',
                    senta_config=senta_config,
                    num_labels=num_labels,
                    is_inference=False)

                sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr)
                sgd_optimizer.minimize(loss)

        if args.verbose:
            lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                (lower_mem, upper_mem, unit))

    if args.do_val:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, loss, accuracy, num_seqs = create_model(
                    args,
                    pyreader_name='test_reader',
                    senta_config=senta_config,
                    num_labels=num_labels,
                    is_inference=False)

        test_prog = test_prog.clone(for_test=True)

    if args.do_infer:
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                infer_pyreader, prop = create_model(
                    args,
                    pyreader_name='infer_reader',
                    senta_config=senta_config,
                    num_labels=num_labels,
                    is_inference=True)
        infer_prog = infer_prog.clone(for_test=True)

    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint:
            init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog)

    elif args.do_val or args.do_infer:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog)

    if args.do_train:
        train_exe = exe
        train_pyreader.decorate_paddle_reader(train_data_generator)
    else:
        train_exe = None
    if args.do_val or args.do_infer:
        test_exe = exe

    if args.do_train:
        train_pyreader.start()
        steps = 0
        total_cost, total_acc, total_num_seqs = [], [], []
        time_begin = time.time()
        while True:
            try:
                steps += 1
                #print("steps...")
                if steps % args.skip_steps == 0:
                    fetch_list = [loss.name, accuracy.name, num_seqs.name]
                else:
                    fetch_list = []

                outputs = train_exe.run(program=train_program, fetch_list=fetch_list, return_numpy=False)
                #print("finished one step")
                if steps % args.skip_steps == 0:
                    np_loss, np_acc, np_num_seqs = outputs
                    np_loss = np.array(np_loss)
                    np_acc = np.array(np_acc)
                    np_num_seqs = np.array(np_num_seqs)
                    total_cost.extend(np_loss * np_num_seqs)
                    total_acc.extend(np_acc * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size()
                        print(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("step: %d, ave loss: %f, "
                        "ave acc: %f, speed: %f steps/s" %
                        (steps, np.sum(total_cost) / np.sum(total_num_seqs),
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
                        print("do evalatation")
                        test_pyreader.decorate_paddle_reader(
                            processor.data_generator(
                                batch_size=args.batch_size,
                                phase='dev',
                                epoch=1,
                                shuffle=False))

                        evaluate(exe, test_prog, test_pyreader,
                                [loss.name, accuracy.name, num_seqs.name],
                                "dev")

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        test_pyreader.decorate_paddle_reader(
            processor.data_generator(
                batch_size=args.batch_size, phase='dev', epoch=1,
                shuffle=False))
        print("Final validation result:")
        evaluate(exe, test_prog, test_pyreader,
            [loss.name, accuracy.name, num_seqs.name], "dev")

        test_pyreader.decorate_paddle_reader(
            processor.data_generator(
                batch_size=args.batch_size, phase='infer', epoch=1,
                shuffle=False))
        evaluate(exe, test_prog, test_pyreader,
            [loss.name, accuracy.name, num_seqs.name], "infer")


    # final eval on test set
    if args.do_infer:
        infer_pyreader.decorate_paddle_reader(
            processor.data_generator(
                batch_size=args.batch_size,
                phase='infer',
                epoch=1,
                shuffle=False))
        print("Final test result:")
        inference(exe, infer_prog, infer_pyreader,
            [prop.name], "infer")

if __name__ == "__main__":
    print_arguments(args)
    main(args)
