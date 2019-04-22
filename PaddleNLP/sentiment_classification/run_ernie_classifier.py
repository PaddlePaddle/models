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

import paddle
import paddle.fluid as fluid

sys.path.append("../models/classification/")
sys.path.append("..")
print(sys.path)

from nets import bow_net
from nets import lstm_net
from nets import cnn_net
from nets import bilstm_net
from nets import gru_net
from nets import ernie_base_net
from nets import ernie_bilstm_net
from preprocess.ernie import task_reader
from models.representation.ernie import ErnieConfig
from models.representation.ernie import ernie_encoder
from models.representation.ernie import ernie_pyreader
from utils import ArgumentGroup
from utils import print_arguments
from utils import init_checkpoint

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("ernie_config_path",         str,  None,           "Path to the json file for ernie model config.")
model_g.add_arg("senta_config_path", str, None, "Path to the json file for senta model config.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints")
model_g.add_arg("model_type", str, "ernie_base", "Type of current ernie model")

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
data_g.add_arg("num_labels",  int,  2,     "label number")
data_g.add_arg("max_seq_len", int,  512,   "Number of words of the longest seqence.")
data_g.add_arg("train_set",           str,  None,  "Path to training data.")
data_g.add_arg("test_set",            str,  None,  "Path to test data.")
data_g.add_arg("dev_set",             str,  None,  "Path to validation data.")
data_g.add_arg("label_map_config",    str,  None,  "label_map_path.")
data_g.add_arg("do_lower_case",       bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation.")
run_type_g.add_arg("do_infer", bool, True, "Whether to perform inference.")

args = parser.parse_args()
# yapf: enable.

def create_model(args,
                 embeddings,
                 labels,
                 is_prediction=False):

    """
    Create Model for sentiment classification based on ERNIE encoder
    """
    sentence_embeddings = embeddings["sentence_embeddings"]
    token_embeddings = embeddings["token_embeddings"]

    if args.model_type == "ernie_base":
        ce_loss, probs = ernie_base_net(sentence_embeddings, labels,
            args.num_labels)

    elif args.model_type == "ernie_bilstm":
        ce_loss, probs = ernie_bilstm_net(token_embeddings, labels,
            args.num_labels)

    else:
        raise ValueError("Unknown network type!")

    if is_prediction:
        return probs
    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    return loss, accuracy, num_seqs


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


def infer(exe, infer_program, infer_pyreader, fetch_list, infer_phase):
    """
    Inference Function
    """
    infer_pyreader.start()
    time_begin = time.time()
    while True:
        try:
            batch_probs = exe.run(program=infer_program, fetch_list=fetch_list,
                                return_numpy=True)
            for probs in batch_probs[0]:
                print("%d\t%f\t%f" % (np.argmax(probs), probs[0], probs[1]))
        except fluid.core.EOFException:
            infer_pyreader.reset()
            break
    time_end = time.time()
    print("[%s] elapsed time: %f s" % (infer_phase, time_end - time_begin))


def main(args):
    """
    Main Function
    """
    args = parser.parse_args()
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    reader = task_reader.ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        random_seed=args.random_seed)

    if not (args.do_train or args.do_val or args.do_infer):
        raise ValueError("For args `do_train`, `do_val` and `do_infer`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                # create ernie_pyreader
                train_pyreader, ernie_inputs, labels = ernie_pyreader(
                    args,
                    pyreader_name='train_reader')

                # get ernie_embeddings
                embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)

                # user defined model based on ernie embeddings
                loss, accuracy, num_seqs = create_model(
                args,
                embeddings,
                labels=labels,
                is_prediction=False)

                """
                sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.lr)
                sgd_optimizer.minimize(loss)
                """
                optimizer = fluid.optimizer.Adam(learning_rate=args.lr)
                optimizer.minimize(loss)

        if args.verbose:
            lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                (lower_mem, upper_mem, unit))

    if args.do_val:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                # create ernie_pyreader
                test_pyreader, ernie_inputs, labels = ernie_pyreader(
                    args,
                    pyreader_name='eval_reader')

                # get ernie_embeddings
                embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)

                # user defined model based on ernie embeddings
                loss, accuracy, num_seqs = create_model(
                args,
                embeddings,
                labels=labels,
                is_prediction=False)

        test_prog = test_prog.clone(for_test=True)

    if args.do_infer:
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                infer_pyreader, ernie_inputs, labels = ernie_pyreader(
                    args,
                    pyreader_name="infer_reader")

                # get ernie_embeddings
                embeddings = ernie_encoder(ernie_inputs, ernie_config=ernie_config)

                probs = create_model(args,
                                    embeddings,
                                    labels=labels,
                                    is_prediction=True)

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
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_iteration_per_drop_scope = 1
        
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            main_program=train_program)
        
        train_pyreader.decorate_tensor_provider(train_data_generator)
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

                outputs = train_exe.run(fetch_list=fetch_list, return_numpy=False)
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
                        test_pyreader.decorate_tensor_provider(
                            reader.data_generator(
                                input_file=args.dev_set,
                                batch_size=args.batch_size,
                                phase='dev',
                                epoch=1,
                                shuffle=False))

                        evaluate(exe, test_prog, test_pyreader,
                                [loss.name, accuracy.name, num_seqs.name],
                                "dev")
                        
                        test_pyreader.decorate_tensor_provider(
                            reader.data_generator(
                                input_file=args.test_set,
                                batch_size=args.batch_size,
                                phase='infer',
                                epoch=1,
                                shuffle=False))
                        
                        evaluate(exe, test_prog, test_pyreader,
                                 [loss.name, accuracy.name, num_seqs.name],
                                 "infer")

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                input_file=args.dev_set,
                batch_size=args.batch_size, phase='dev', epoch=1,
                shuffle=False))
        print("Final validation result:")
        evaluate(exe, test_prog, test_pyreader,
            [loss.name, accuracy.name, num_seqs.name], "dev")

    # final eval on test set
    if args.do_infer:
        infer_pyreader.decorate_tensor_provider(
            reader.data_generator(
                input_file=args.test_set,
                batch_size=args.batch_size,
                phase='infer',
                epoch=1,
                shuffle=False))
        print("Final test result:")
        infer(exe, infer_prog, infer_pyreader,
            [probs.name], "infer")

if __name__ == "__main__":
    print_arguments(args)
    main(args)
