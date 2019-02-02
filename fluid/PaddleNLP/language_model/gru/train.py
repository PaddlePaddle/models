import os
import sys
import time
import six

import numpy as np
import math
import argparse
import paddle.fluid as fluid
import paddle

import utils

SEED = 102


def parse_args():
    parser = argparse.ArgumentParser("language_model benchmark.")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run \
        the task with continuous evaluation logs.')
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    args = parser.parse_args()
    return args


def network(src, dst, vocab_size, hid_size, init_low_bound, init_high_bound):
    """ network definition """
    emb_lr_x = 10.0
    gru_lr_x = 1.0
    fc_lr_x = 1.0
    emb = fluid.layers.embedding(
        input=src,
        size=[vocab_size, hid_size],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=init_low_bound, high=init_high_bound),
            learning_rate=emb_lr_x),
        is_sparse=True)

    fc0 = fluid.layers.fc(input=emb,
                          size=hid_size * 3,
                          param_attr=fluid.ParamAttr(
                              initializer=fluid.initializer.Uniform(
                                  low=init_low_bound, high=init_high_bound),
                              learning_rate=gru_lr_x))
    gru_h0 = fluid.layers.dynamic_gru(
        input=fc0,
        size=hid_size,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=init_low_bound, high=init_high_bound),
            learning_rate=gru_lr_x))

    fc = fluid.layers.fc(input=gru_h0,
                         size=vocab_size,
                         act='softmax',
                         param_attr=fluid.ParamAttr(
                             initializer=fluid.initializer.Uniform(
                                 low=init_low_bound, high=init_high_bound),
                             learning_rate=fc_lr_x))

    cost = fluid.layers.cross_entropy(input=fc, label=dst)
    return cost


def train(train_reader,
          vocab,
          network,
          hid_size,
          base_lr,
          batch_size,
          pass_num,
          use_cuda,
          parallel,
          model_dir,
          init_low_bound=-0.04,
          init_high_bound=0.04):
    """ train network """

    args = parse_args()
    if args.enable_ce:
        # random seed must set before configuring the network.
        fluid.default_startup_program().random_seed = SEED
    vocab_size = len(vocab)

    #Input data
    src_wordseq = fluid.layers.data(
        name="src_wordseq", shape=[1], dtype="int64", lod_level=1)
    dst_wordseq = fluid.layers.data(
        name="dst_wordseq", shape=[1], dtype="int64", lod_level=1)

    # Train program
    avg_cost = None
    cost = network(src_wordseq, dst_wordseq, vocab_size, hid_size,
                   init_low_bound, init_high_bound)
    avg_cost = fluid.layers.mean(x=cost)

    # Optimization to minimize lost
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=base_lr,
            decay_steps=2100 * 4,
            decay_rate=0.5,
            staircase=True))
    sgd_optimizer.minimize(avg_cost)

    # Initialize executor
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=avg_cost.name)

    total_time = 0.0
    fetch_list = [avg_cost.name]
    for pass_idx in six.moves.xrange(pass_num):
        epoch_idx = pass_idx + 1
        print("epoch_%d start" % epoch_idx)

        t0 = time.time()
        i = 0
        newest_ppl = 0
        for data in train_reader():
            i += 1
            lod_src_wordseq = utils.to_lodtensor([dat[0] for dat in data],
                                                 place)
            lod_dst_wordseq = utils.to_lodtensor([dat[1] for dat in data],
                                                 place)
            ret_avg_cost = train_exe.run(feed={
                "src_wordseq": lod_src_wordseq,
                "dst_wordseq": lod_dst_wordseq
            },
                                         fetch_list=fetch_list)
            avg_ppl = np.exp(ret_avg_cost[0])
            newest_ppl = np.mean(avg_ppl)
            if i % 100 == 0:
                print("step:%d ppl:%.3f" % (i, newest_ppl))

        t1 = time.time()
        total_time += t1 - t0
        print("epoch:%d num_steps:%d time_cost(s):%f" %
              (epoch_idx, i, total_time / epoch_idx))

        if pass_idx == pass_num - 1 and args.enable_ce:
            #Note: The following logs are special for CE monitoring.
            #Other situations do not need to care about these logs.
            gpu_num = get_cards(args)
            if gpu_num == 1:
                print("kpis	imikolov_20_pass_duration	%s" %
                      (total_time / epoch_idx))
                print("kpis	imikolov_20_avg_ppl	%s" % newest_ppl)
            else:
                print("kpis	imikolov_20_pass_duration_card%s	%s" % \
                      (gpu_num, total_time / epoch_idx))
                print("kpis	imikolov_20_avg_ppl_card%s	%s" %
                      (gpu_num, newest_ppl))
        save_dir = "%s/epoch_%d" % (model_dir, epoch_idx)
        feed_var_names = ["src_wordseq", "dst_wordseq"]
        fetch_vars = [avg_cost]
        fluid.io.save_inference_model(save_dir, feed_var_names, fetch_vars, exe)
        print("model saved in %s" % save_dir)

    print("finish training")


def get_cards(args):
    if args.enable_ce:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        num = len(cards.split(","))
        return num
    else:
        return args.num_devices


def train_net():
    """ do training """
    batch_size = 20
    args = parse_args()
    vocab, train_reader, test_reader = utils.prepare_data(
        batch_size=batch_size * get_cards(args), buffer_size=1000, \
        word_freq_threshold=0, enable_ce = args.enable_ce)
    train(
        train_reader=train_reader,
        vocab=vocab,
        network=network,
        hid_size=200,
        base_lr=1.0,
        batch_size=batch_size,
        pass_num=12,
        use_cuda=True,
        parallel=True,
        model_dir="model",
        init_low_bound=-0.1,
        init_high_bound=0.1)


if __name__ == "__main__":
    train_net()
