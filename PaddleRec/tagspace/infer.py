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


def parse_args():
    parser = argparse.ArgumentParser("tagspace benchmark.")
    parser.add_argument(
        '--test_dir', type=str, default='test_data', help='test file address')
    parser.add_argument(
        '--vocab_tag_path',
        type=str,
        default='vocab_tag.txt',
        help='vocab path')
    parser.add_argument(
        '--start_index', type=int, default='1', help='start index')
    parser.add_argument(
        '--last_index', type=int, default='10', help='end index')
    parser.add_argument(
        '--model_dir', type=str, default='model_', help='model dir')
    parser.add_argument(
        '--use_cuda', type=int, default='0', help='whether use cuda')
    parser.add_argument(
        '--batch_size', type=int, default='5', help='batch_size')
    args = parser.parse_args()
    return args


def infer(test_reader, vocab_tag, use_cuda, model_path, epoch):
    """ inference function """
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    with fluid.scope_guard(fluid.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            model_path, exe)
        t0 = time.time()
        step_id = 0
        true_num = 0
        all_num = 0
        size = vocab_tag
        value = []
        print("epoch " + str(epoch) + " start")
        for data in test_reader():
            step_id += 1
            lod_text_seq = utils.to_lodtensor([dat[0] for dat in data], place)
            lod_tag = utils.to_lodtensor([dat[1] for dat in data], place)
            lod_pos_tag = utils.to_lodtensor([dat[2] for dat in data], place)
            para = exe.run(infer_program,
                           feed={"text": lod_text_seq,
                                 "pos_tag": lod_tag},
                           fetch_list=fetch_vars,
                           return_numpy=False)
            value.append(para[0]._get_float_element(0))
            if step_id % size == 0 and step_id > 1:
                all_num += 1
                true_pos = [dat[2] for dat in data][0][0]
                if value.index(max(value)) == int(true_pos):
                    true_num += 1
                value = []
        print("epoch:" + str(epoch) + "\tacc:" + str(1.0 * true_num / all_num))
        t1 = time.time()


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
    vocab_tag_path = args.vocab_tag_path
    use_cuda = True if args.use_cuda else False
    print("start index: ", start_index, " last_index:", last_index)
    vocab_text, vocab_tag, test_reader = utils.prepare_data(
        test_dir,
        "",
        vocab_tag_path,
        batch_size=1,
        neg_size=0,
        buffer_size=1000,
        is_train=False)

    for epoch in range(start_index, last_index + 1):
        epoch_path = model_dir + "/epoch_" + str(epoch)
        infer(
            test_reader=test_reader,
            vocab_tag=vocab_tag,
            use_cuda=False,
            model_path=epoch_path,
            epoch=epoch)
