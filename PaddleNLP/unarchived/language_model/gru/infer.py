import sys
import time
import math
import unittest
import contextlib
import numpy as np
import six

import paddle
import paddle.fluid as fluid

import utils


def infer(test_reader, use_cuda, model_path):
    """ inference function """
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    with fluid.scope_guard(fluid.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            model_path, exe)

        accum_cost = 0.0
        accum_words = 0
        t0 = time.time()
        for data in test_reader():
            src_wordseq = utils.to_lodtensor([dat[0] for dat in data], place)
            dst_wordseq = utils.to_lodtensor([dat[1] for dat in data], place)
            avg_cost = exe.run(
                infer_program,
                feed={"src_wordseq": src_wordseq,
                      "dst_wordseq": dst_wordseq},
                fetch_list=fetch_vars)

            nwords = src_wordseq.lod()[0][-1]

            cost = np.array(avg_cost) * nwords
            accum_cost += cost
            accum_words += nwords

        ppl = math.exp(accum_cost / accum_words)
        t1 = time.time()
        print("model:%s ppl:%.3f time_cost(s):%.2f" %
              (model_path, ppl, t1 - t0))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s model_dir start_epoch last_epoch(inclusive)")
        exit(0)

    model_dir = sys.argv[1]
    try:
        start_index = int(sys.argv[2])
        last_index = int(sys.argv[3])
    except:
        print("Usage: %s model_dir start_epoch last_epoch(inclusive)")
        exit(-1)

    vocab, train_reader, test_reader = utils.prepare_data(
        batch_size=20, buffer_size=1000, word_freq_threshold=0)

    for epoch in six.moves.xrange(start_index, last_index + 1):
        epoch_path = model_dir + "/epoch_" + str(epoch)
        infer(test_reader=test_reader, use_cuda=True, model_path=epoch_path)
