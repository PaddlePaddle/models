import sys
import argparse
import time
import math
import unittest
import contextlib
import numpy as np
import six
import paddle.fluid as fluid
import paddle
import utils
import nets as net


def parse_args():
    parser = argparse.ArgumentParser("ssr benchmark.")
    parser.add_argument(
        '--test_dir', type=str, default='test_data', help='test file address')
    parser.add_argument(
        '--vocab_path', type=str, default='vocab.txt', help='vocab path')
    parser.add_argument(
        '--start_index', type=int, default='1', help='start index')
    parser.add_argument(
        '--last_index', type=int, default='10', help='end index')
    parser.add_argument(
        '--model_dir', type=str, default='model_output', help='model dir')
    parser.add_argument(
        '--use_cuda', type=int, default='0', help='whether use cuda')
    parser.add_argument(
        '--batch_size', type=int, default='50', help='batch_size')
    parser.add_argument(
        '--hid_size', type=int, default='128', help='hidden size')
    parser.add_argument(
        '--emb_size', type=int, default='128', help='embedding size')
    args = parser.parse_args()
    return args


def model(vocab_size, emb_size, hidden_size):
    user_data = fluid.layers.data(
        name="user", shape=[1], dtype="int64", lod_level=1)
    all_item_data = fluid.layers.data(
        name="all_item", shape=[vocab_size, 1], dtype="int64")

    user_emb = fluid.layers.embedding(
        input=user_data, size=[vocab_size, emb_size], param_attr="emb.item")
    all_item_emb = fluid.layers.embedding(
        input=all_item_data, size=[vocab_size, emb_size], param_attr="emb.item")
    all_item_emb_re = fluid.layers.reshape(x=all_item_emb, shape=[-1, emb_size])

    user_encoder = net.GrnnEncoder(hidden_size=hidden_size)
    user_enc = user_encoder.forward(user_emb)
    user_hid = fluid.layers.fc(input=user_enc,
                               size=hidden_size,
                               param_attr='user.w',
                               bias_attr="user.b")
    user_exp = fluid.layers.expand(x=user_hid, expand_times=[1, vocab_size])
    user_re = fluid.layers.reshape(x=user_exp, shape=[-1, hidden_size])

    all_item_hid = fluid.layers.fc(input=all_item_emb_re,
                                   size=hidden_size,
                                   param_attr='item.w',
                                   bias_attr="item.b")
    cos_item = fluid.layers.cos_sim(X=all_item_hid, Y=user_re)
    all_pre_ = fluid.layers.reshape(x=cos_item, shape=[-1, vocab_size])
    pos_label = fluid.layers.data(name="pos_label", shape=[1], dtype="int64")
    acc = fluid.layers.accuracy(input=all_pre_, label=pos_label, k=20)
    return acc


def infer(args, vocab_size, test_reader):
    """ inference function """
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    emb_size = args.emb_size
    hid_size = args.hid_size
    batch_size = args.batch_size
    model_path = args.model_dir
    with fluid.scope_guard(fluid.Scope()):
        main_program = fluid.Program()
        start_up_program = fluid.Program()
        with fluid.program_guard(main_program, start_up_program):
            acc = model(vocab_size, emb_size, hid_size)
            for epoch in range(start_index, last_index + 1):
                copy_program = main_program.clone()
                model_path = model_dir + "/epoch_" + str(epoch)
                fluid.io.load_params(
                    executor=exe, dirname=model_path, main_program=copy_program)
                accum_num_recall = 0.0
                accum_num_sum = 0.0
                t0 = time.time()
                step_id = 0
                for data in test_reader():
                    step_id += 1
                    user_data, pos_label = utils.infer_data(data, place)
                    all_item_numpy = np.tile(
                        np.arange(vocab_size), len(pos_label)).reshape(
                            len(pos_label), vocab_size, 1)
                    para = exe.run(copy_program,
                                   feed={
                                       "user": user_data,
                                       "all_item": all_item_numpy,
                                       "pos_label": pos_label
                                   },
                                   fetch_list=[acc.name],
                                   return_numpy=False)

                    acc_ = para[0]._get_float_element(0)
                    data_length = len(
                        np.concatenate(
                            pos_label, axis=0).astype("int64"))
                    accum_num_sum += (data_length)
                    accum_num_recall += (data_length * acc_)
                    if step_id % 1 == 0:
                        print("step:%d  " % (step_id),
                              accum_num_recall / accum_num_sum)
                t1 = time.time()
                print("model:%s recall@20:%.3f time_cost(s):%.2f" %
                      (model_path, accum_num_recall / accum_num_sum, t1 - t0))


if __name__ == "__main__":
    args = parse_args()
    start_index = args.start_index
    last_index = args.last_index
    test_dir = args.test_dir
    model_dir = args.model_dir
    batch_size = args.batch_size
    vocab_path = args.vocab_path
    use_cuda = True if args.use_cuda else False
    print("start index: ", start_index, " last_index:", last_index)
    test_reader, vocab_size = utils.construct_test_data(
        test_dir, vocab_path, batch_size=args.batch_size)
    infer(args, vocab_size, test_reader=test_reader)
