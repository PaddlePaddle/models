import sys
import time
import math
import unittest
import contextlib
import numpy as np
import six
import paddle.fluid as fluid
import paddle

import utils


def infer(test_reader, use_cuda, model_path):
    """ inference function """
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    with fluid.scope_guard(fluid.core.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            model_path, exe)
        accum_num_recall = 0.0
        accum_num_sum = 0.0
        t0 = time.time()
        step_id = 0
        for data in test_reader():
            step_id += 1
            src_wordseq = utils.to_lodtensor([dat[0] for dat in data], place)
            label_data = [dat[1] for dat in data]
            dst_wordseq = utils.to_lodtensor(label_data, place)
            para = exe.run(
                infer_program,
                feed={"src_wordseq": src_wordseq,
                      "dst_wordseq": dst_wordseq},
                fetch_list=fetch_vars,
                return_numpy=False)

            acc_ = para[1]._get_float_element(0)
            data_length = len(
                np.concatenate(
                    label_data, axis=0).astype("int64"))
            accum_num_sum += (data_length)
            accum_num_recall += (data_length * acc_)
            if step_id % 100 == 0:
                print("step:%d  " % (step_id), accum_num_recall / accum_num_sum)
        t1 = time.time()
        print("model:%s recall@20:%.3f time_cost(s):%.2f" %
              (model_path, accum_num_recall / accum_num_sum, t1 - t0))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            "Usage: %s model_dir start_epoch last_epoch(inclusive) train_file test_file"
        )
        exit(0)
    train_file = ""
    test_file = ""
    model_dir = sys.argv[1]
    try:
        start_index = int(sys.argv[2])
        last_index = int(sys.argv[3])
        train_file = sys.argv[4]
        test_file = sys.argv[5]
    except:
        iprint(
            "Usage: %s model_dir start_ipoch last_epoch(inclusive) train_file test_file"
        )
        exit(-1)
    vocab, train_reader, test_reader = utils.prepare_data(
        train_file,
        test_file,
        batch_size=5,
        buffer_size=1000,
        word_freq_threshold=0)

    for epoch in xrange(start_index, last_index + 1):
        epoch_path = model_dir + "/epoch_" + str(epoch)
        infer(test_reader=test_reader, use_cuda=True, model_path=epoch_path)
