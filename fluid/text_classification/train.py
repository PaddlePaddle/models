import os
import sys
import time
import unittest
import contextlib
import numpy as np

import paddle.fluid as fluid
import paddle.v2 as paddle

import utils
from nets import bow_net
from nets import cnn_net
from nets import lstm_net
from nets import gru_net


def train(train_reader,
          word_dict,
          network,
          use_cuda,
          parallel,
          save_dirname,
          lr=0.2,
          batch_size=128,
          pass_num=30):
    """
    train network
    """

    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        train_py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1, 1], [-1, 1]],
            lod_levels=[1, 0],
            dtypes=["int64", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            data, label = fluid.layers.read_file(train_py_reader)
            cost, acc, prediction = network(data, label, len(word_dict))
            sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
            sgd_optimizer.minimize(cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    train_py_reader.decorate_paddle_reader(train_reader)

    if parallel:
        parallel_exe = fluid.ParallelExecutor(
            main_program=train_prog, use_cuda=use_cuda, loss_name=cost.name)

    train_fetch_list = [cost.name, acc.name]

    for pass_id in xrange(pass_num):
        train_py_reader.start()
        train_info = [[], []]
        batch_id = 0
        try:
            while True:
                if not parallel:
                    avg_cost_np, avg_acc_np = exe.run(
                        train_prog, fetch_list=train_fetch_list)
                else:
                    avg_cost_np, avg_acc_np = parallel_exe.run(
                        fetch_list=train_fetch_list)
                avg_cost_np = np.mean(np.array(avg_cost_np))
                avg_acc_np = np.mean(np.array(avg_acc_np))
                train_info[0].append(avg_cost_np)
                train_info[1].append(avg_acc_np)
                batch_id += 1
        except fluid.core.EOFException:
            train_py_reader.reset()

        avg_cost = np.array(train_info[0]).mean()
        avg_acc = np.array(train_info[1]).mean()

        print("pass_id: %d, avg_acc: %f, avg_cost: %f" %
              (pass_id, avg_cost, avg_acc))
        sys.stdout.flush()

        model_path = os.path.join(save_dirname + "/" + "epoch" + str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_inference_model(
            model_path, [data.name, label.name],
            acc,
            exe,
            main_program=train_prog)


def train_net():
    word_dict, train_reader, test_reader = utils.prepare_data(
        "imdb", self_dict=False, batch_size=128, buf_size=50000)

    if sys.argv[1] == "bow":
        train(
            train_reader,
            word_dict,
            bow_net,
            use_cuda=False,
            parallel=False,
            save_dirname="bow_model",
            lr=0.002,
            pass_num=30,
            batch_size=128)
    elif sys.argv[1] == "cnn":
        train(
            train_reader,
            word_dict,
            cnn_net,
            use_cuda=True,
            parallel=False,
            save_dirname="cnn_model",
            lr=0.01,
            pass_num=30,
            batch_size=4)
    elif sys.argv[1] == "lstm":
        train(
            train_reader,
            word_dict,
            lstm_net,
            use_cuda=True,
            parallel=False,
            save_dirname="lstm_model",
            lr=0.05,
            pass_num=30,
            batch_size=4)
    elif sys.argv[1] == "gru":
        train(
            train_reader,
            word_dict,
            lstm_net,
            use_cuda=True,
            parallel=False,
            save_dirname="gru_model",
            lr=0.05,
            pass_num=30,
            batch_size=128)
    else:
        print("network name cannot be found!")
        sys.exit(1)


if __name__ == "__main__":
    train_net()
