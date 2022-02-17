import argparse
import sys
import time
import math
import unittest
import contextlib
import numpy as np
import six
import paddle.fluid as fluid
import paddle
import net
import utils


def parse_args():
    parser = argparse.ArgumentParser("gru4rec benchmark.")
    parser.add_argument(
        '--test_dir', type=str, default='test_data', help='test file address')
    parser.add_argument(
        '--start_index', type=int, default='1', help='start index')
    parser.add_argument(
        '--last_index', type=int, default='3', help='last index')
    parser.add_argument(
        '--model_dir', type=str, default='model_neg_recall20', help='model dir')
    parser.add_argument(
        '--use_cuda', type=int, default='0', help='whether use cuda')
    parser.add_argument(
        '--batch_size', type=int, default='5', help='batch_size')
    parser.add_argument(
        '--hid_size', type=int, default='100', help='batch_size')
    parser.add_argument(
        '--vocab_path', type=str, default='vocab.txt', help='vocab file')
    args = parser.parse_args()
    return args


def infer(args, vocab_size, test_reader, use_cuda):
    """ inference function """
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    hid_size = args.hid_size
    batch_size = args.batch_size
    with fluid.scope_guard(fluid.Scope()):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            acc = net.infer_network(vocab_size, batch_size, hid_size)
            for epoch in range(start_index, last_index + 1):
                copy_program = main_program.clone()
                model_path = model_dir + "/epoch_" + str(epoch)
                fluid.load(copy_program, model_path, exe)
                accum_num_recall = 0.0
                accum_num_sum = 0.0
                t0 = time.time()
                step_id = 0
                for data in test_reader():
                    step_id += 1
                    label_data = [dat[1] for dat in data]
                    ls, lp = utils.to_lodtensor_bpr_test(data, vocab_size,
                                                         place)
                    para = exe.run(
                        copy_program,
                        feed={
                            "src": ls,
                            "all_label":
                            np.arange(vocab_size).reshape(vocab_size, 1),
                            "pos_label": lp
                        },
                        fetch_list=[acc.name],
                        return_numpy=False)

                    acc_ = np.array(para[0])[0]
                    data_length = len(
                        np.concatenate(
                            label_data, axis=0).astype("int64"))
                    accum_num_sum += (data_length)
                    accum_num_recall += (data_length * acc_)
                    if step_id % 1 == 0:
                        print("step:%d  recall@20:%.4f" %
                              (step_id, accum_num_recall / accum_num_sum))
                t1 = time.time()
                print("model:%s recall@20:%.4f time_cost(s):%.2f" %
                      (model_path, accum_num_recall / accum_num_sum, t1 - t0))


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    utils.check_version()
    args = parse_args()
    start_index = args.start_index
    last_index = args.last_index
    test_dir = args.test_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    vocab_path = args.vocab_path
    use_cuda = True if args.use_cuda else False
    print("start index: ", start_index, " last_index:", last_index)
    vocab_size, test_reader = utils.prepare_data(
        test_dir,
        vocab_path,
        batch_size=batch_size,
        buffer_size=1000,
        word_freq_threshold=0,
        is_train=False)

    infer(args, vocab_size, test_reader=test_reader, use_cuda=use_cuda)
