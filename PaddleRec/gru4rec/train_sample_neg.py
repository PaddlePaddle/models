import os
import sys
import time
import six
import numpy as np
import math
import argparse
import paddle.fluid as fluid
import paddle
import time
import utils
import net

SEED = 102


def parse_args():
    parser = argparse.ArgumentParser("gru4rec benchmark.")
    parser.add_argument(
        '--train_dir', type=str, default='train_data', help='train file')
    parser.add_argument(
        '--vocab_path', type=str, default='vocab.txt', help='vocab file')
    parser.add_argument(
        '--is_local', type=int, default=1, help='whether is local')
    parser.add_argument(
        '--hid_size', type=int, default=100, help='hidden-dim size')
    parser.add_argument(
        '--neg_size', type=int, default=10, help='neg item size')
    parser.add_argument(
        '--loss', type=str, default="bpr", help='loss: bpr/cross_entropy')
    parser.add_argument(
        '--model_dir', type=str, default='model_neg_recall20', help='model dir')
    parser.add_argument(
        '--batch_size', type=int, default=5, help='num of batch size')
    parser.add_argument(
        '--print_batch', type=int, default=10, help='num of print batch')
    parser.add_argument(
        '--pass_num', type=int, default=10, help='number of epoch')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether use gpu')
    parser.add_argument(
        '--parallel', type=int, default=0, help='whether parallel')
    parser.add_argument(
        '--base_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    args = parser.parse_args()
    return args


def get_cards(args):
    return args.num_devices


def train():
    """ do training """
    args = parse_args()
    hid_size = args.hid_size
    train_dir = args.train_dir
    vocab_path = args.vocab_path
    use_cuda = True if args.use_cuda else False
    parallel = True if args.parallel else False
    print("use_cuda:", use_cuda, "parallel:", parallel)
    batch_size = args.batch_size
    vocab_size, train_reader = utils.prepare_data(
        train_dir, vocab_path, batch_size=batch_size * get_cards(args),\
        buffer_size=1000, word_freq_threshold=0, is_train=True)

    # Train program
    if args.loss == 'bpr':
        print('bpr loss')
        src, pos_label, label, avg_cost = net.train_bpr_network(
            neg_size=args.neg_size, vocab_size=vocab_size, hid_size=hid_size)
    else:
        print('cross-entory loss')
        src, pos_label, label, avg_cost = net.train_cross_entropy_network(
            neg_size=args.neg_size, vocab_size=vocab_size, hid_size=hid_size)

    # Optimization to minimize lost
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=args.base_lr)
    sgd_optimizer.minimize(avg_cost)

    # Initialize executor
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    if parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda, loss_name=avg_cost.name)
    else:
        train_exe = exe

    pass_num = args.pass_num
    model_dir = args.model_dir
    fetch_list = [avg_cost.name]

    total_time = 0.0
    for pass_idx in six.moves.xrange(pass_num):
        epoch_idx = pass_idx + 1
        print("epoch_%d start" % epoch_idx)

        t0 = time.time()
        i = 0
        newest_ppl = 0
        for data in train_reader():
            i += 1
            ls, lp, ll = utils.to_lodtensor_bpr(data, args.neg_size, vocab_size,
                                                place)
            ret_avg_cost = train_exe.run(
                feed={"src": ls,
                      "label": ll,
                      "pos_label": lp},
                fetch_list=fetch_list)
            avg_ppl = np.exp(ret_avg_cost[0])
            newest_ppl = np.mean(avg_ppl)
            if i % args.print_batch == 0:
                print("step:%d ppl:%.3f" % (i, newest_ppl))

        t1 = time.time()
        total_time += t1 - t0
        print("epoch:%d num_steps:%d time_cost(s):%f" %
              (epoch_idx, i, total_time / epoch_idx))
        save_dir = "%s/epoch_%d" % (model_dir, epoch_idx)
        fluid.save(fluid.default_main_program(), model_path=save_dir)
        print("model saved in %s" % save_dir)

    print("finish training")


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    utils.check_version()
    train()
