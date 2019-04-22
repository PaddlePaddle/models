"""
Emotion Detection Task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import multiprocessing
import sys
sys.path.append("../")

import paddle
import paddle.fluid as fluid
import numpy as np

from models.classification import nets
import reader
import config
import utils

parser = argparse.ArgumentParser(__doc__)
model_g = utils.ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("config_path", str, None, "Path to the json file for EmoTect model config.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("output_dir", str, None, "Directory path to save checkpoints")

train_g = utils.ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
train_g.add_arg("lr", float, 0.002, "The Learning rate value for training.")

log_g = utils.ArgumentGroup(parser, "logging", "logging related")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log")

data_g = utils.ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("data_dir", str, None, "Directory path to training data.")
data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
data_g.add_arg("batch_size", int, 256, "Total examples' number in batch for training.")
data_g.add_arg("random_seed", int, 0, "Random seed.")

run_type_g = utils.ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")
run_type_g.add_arg("task_name", str, None, "The name of task to perform sentiment classification.")
run_type_g.add_arg("do_train", bool, False, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, False, "Whether to perform evaluation.")
run_type_g.add_arg("do_infer", bool, False, "Whether to perform inference.")

args = parser.parse_args()

def create_model(args,
                 pyreader_name,
                 emotect_config,
                 num_labels,
                 is_infer=False):
    """
    Create Model for sentiment classification
    """
    if is_infer:
        pyreader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1, 1]],
            dtypes=['int64'],
            lod_levels=[1],
            name=pyreader_name,
            use_double_buffer=False)
    else:
        pyreader = fluid.layers.py_reader(
            capacity=16,
            shapes=([-1, 1], [-1, 1]),
            dtypes=('int64', 'int64'),
            lod_levels=(1, 0),
            name=pyreader_name,
            use_double_buffer=False)

    if emotect_config['model_type'] == "cnn_net":
        network = nets.cnn_net
    elif emotect_config['model_type'] == "bow_net":
        network = nets.bow_net
    elif emotect_config['model_type'] == "lstm_net":
        network = nets.lstm_net
    elif emotect_config['model_type'] == "bilstm_net":
        network = nets.bilstm_net
    elif emotect_config['model_type'] == "gru_net":
        network = nets.gru_net
    elif emotect_config['model_type'] == "textcnn_net":
        network = nets.textcnn_net
    else:
        raise ValueError("Unknown network type!")

    if is_infer:
        data = fluid.layers.read_file(pyreader)
        probs = network(data, None, emotect_config["vocab_size"], class_dim=num_labels, is_infer=True)
        return pyreader, probs

    data, label = fluid.layers.read_file(pyreader)
    avg_loss, probs = network(data, label, emotect_config["vocab_size"], class_dim=num_labels)
    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=label, total=num_seqs)
    return pyreader, avg_loss, accuracy, num_seqs


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
    print("[%s evaluation] avg loss: %f, avg acc: %f, elapsed time: %f s" %
        (eval_phase, np.sum(total_cost) / np.sum(total_num_seqs),
        np.sum(total_acc) / np.sum(total_num_seqs), time_end - time_begin))


def infer(exe, infer_program, infer_pyreader, fetch_list, infer_phase):
    infer_pyreader.start()
    time_begin = time.time()
    while True:
        try:
            batch_probs = exe.run(program=infer_program,
                            fetch_list=fetch_list,
                            return_numpy=True)
            for probs in batch_probs[0]:
                print("%d\t%f\t%f\t%f" % (np.argmax(probs), probs[0], probs[1], probs[2]))
        except fluid.core.EOFException as e:
            infer_pyreader.reset()
            break
    time_end = time.time()
    print("[%s] elapsed time: %f s" % (infer_phase, time_end - time_begin))


def main(args):
    """
    Main Function
    """
    emotect_config = config.EmoTectConfig(args.config_path)

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
    else:
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    task_name = args.task_name.lower()
    processor = reader.EmoTectProcessor(data_dir=args.data_dir,
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
            epoch=args.epoch)

        num_train_examples = processor.get_num_examples(phase="train")
        max_train_steps = args.epoch * num_train_examples // args.batch_size + 1

        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, loss, accuracy, num_seqs = create_model(
                    args,
                    pyreader_name='train_reader',
                    emotect_config=emotect_config,
                    num_labels=num_labels,
                    is_infer=False)

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
                    emotect_config=emotect_config,
                    num_labels=num_labels,
                    is_infer=False)
        test_prog = test_prog.clone(for_test=True)

    if args.do_infer:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                infer_pyreader, probs = create_model(
                    args,
                    pyreader_name='infer_reader',
                    emotect_config=emotect_config,
                    num_labels=num_labels,
                    is_infer=True)
        test_prog = test_prog.clone(for_test=True)

    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint:
            utils.init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog)
    elif args.do_val or args.do_infer:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or infer!")
        utils.init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=test_prog)

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
                if steps % args.skip_steps == 0:
                    fetch_list = [loss.name, accuracy.name, num_seqs.name]
                else:
                    fetch_list = []

                outputs = train_exe.run(program=train_program,
                                        fetch_list=fetch_list,
                                        return_numpy=False)
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
                    print("step: %d, avg loss: %f, "
                        "avg acc: %f, speed: %f steps/s" %
                        (steps, np.sum(total_cost) / np.sum(total_num_seqs),
                        np.sum(total_acc) / np.sum(total_num_seqs),
                        args.skip_steps / used_time))
                    total_cost, total_acc, total_num_seqs = [], [], []
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    # evaluate on dev set
                    if args.do_val:
                        test_pyreader.decorate_paddle_reader(
                            processor.data_generator(
                                batch_size=args.batch_size,
                                phase='dev',
                                epoch=1))
                        evaluate(test_exe, test_prog, test_pyreader,
                                [loss.name, accuracy.name, num_seqs.name],
                                "dev")

            except fluid.core.EOFException:
                save_path = os.path.join(args.output_dir, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # evaluate on test set
    if not args.do_train and args.do_val:
        test_pyreader.decorate_paddle_reader(
            processor.data_generator(
                batch_size=args.batch_size,
                phase='test',
                epoch=1))
        print("Final test result:")
        evaluate(test_exe, test_prog, test_pyreader,
                [loss.name, accuracy.name, num_seqs.name],
                "test")

    # infer
    if args.do_infer:
        infer_pyreader.decorate_paddle_reader(
            processor.data_generator(
                batch_size=args.batch_size,
                phase='infer',
                epoch=1))
        infer(test_exe, test_prog, infer_pyreader,
             [probs.name], "infer")

if __name__ == "__main__":
    utils.print_arguments(args)
    main(args)
