"""
For http://wiki.baidu.com/display/LegoNet/Text+Classification
"""
import paddle.fluid as fluid
import paddle.v2 as paddle
import numpy as np
import sys
import time
import unittest
import contextlib
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
    data = fluid.layers.data(
        name="words", 
        shape=[1], 
        dtype="int64", 
        lod_level=1)

    label = fluid.layers.data(
        name="label", 
        shape=[1], 
        dtype="int64")

    if not parallel:
        cost, acc, prediction = network(
            data, label, len(word_dict))
    else:
        places = fluid.layers.get_places(device_count = 2)
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            cost, acc, prediction = network(
            pd.read_input(data), 
            pd.read_input(label), 
            len(word_dict))

            pd.write_output(cost)
            pd.write_output(acc)

        cost, acc = pd()
        cost = fluid.layers.mean(cost)
        acc = fluid.layers.mean(acc)

    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    exe.run(fluid.default_startup_program())
    for pass_id in xrange(pass_num):
        avg_cost_list, avg_acc_list = [], []
        for data in train_reader():
            avg_cost_np, avg_acc_np = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[cost, acc])
            avg_cost_list.append(avg_cost_np)
            avg_acc_list.append(avg_acc_np)
        print("pass_id: %d, avg_acc: %f" % (pass_id, np.mean(avg_acc_list)))
    # save_model
    fluid.io.save_inference_model(
            save_dirname, 
            ["words", "label"],
            acc, exe)

def train_net():
    word_dict, train_reader, test_reader = utils.prepare_data(
            "imdb", self_dict = False,
            batch_size = 128, buf_size = 50000)
    
    if sys.argv[1] == "bow":
        train(train_reader, word_dict, bow_net, use_cuda=False,
                parallel=False, save_dirname="bow_model", lr=0.002,
                pass_num=1, batch_size=128)
    elif sys.argv[1] == "cnn":
        train(train_reader, word_dict, cnn_net, use_cuda=True,
                parallel=False, save_dirname="cnn_model", lr=0.01,
                pass_num=30, batch_size=4)
    elif sys.argv[1] == "lstm":
        train(train_reader, word_dict, lstm_net, use_cuda=True,
                parallel=False, save_dirname="lstm_model", lr=0.05,
                pass_num=30, batch_size=4)
    elif sys.argv[1] == "gru":
        train(train_reader, word_dict, bow_net, use_cuda=True,
                parallel=False, save_dirname="gru_model", lr=0.05,
                pass_num=30, batch_size=128)
    else:
        print("network name cannot be found!")
        sys.exit(1)    

if __name__ == "__main__":
    train_net()
