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

def infer(test_reader, vocab_tag, use_cuda, model_path):
    """ inference function """
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    with fluid.scope_guard(fluid.core.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            model_path, exe)
        t0 = time.time()
        step_id = 0
        true_num = 0 
        all_num = 0
        size = len(vocab_tag)
        value = []
        for data in test_reader():
            step_id += 1
            lod_text_seq = utils.to_lodtensor([dat[0] for dat in data], place)
            lod_tag = utils.to_lodtensor([dat[1] for dat in data], place)
            lod_pos_tag = utils.to_lodtensor([dat[2] for dat in data], place)
            para = exe.run(
                infer_program,
                feed={
                    "text": lod_text_seq,
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
            if step_id % 1000 == 0:
                print(step_id, 1.0 * true_num / all_num)
        t1 = time.time()

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
        print(
            "Usage: %s model_dir start_ipoch last_epoch(inclusive) train_file test_file"
        )
        exit(-1)
    vocab_text, vocab_tag, train_reader, test_reader = utils.prepare_data(
        train_file,
        test_file,
        batch_size=1,
        buffer_size=1000,
        word_freq_threshold=0)

    for epoch in xrange(start_index, last_index + 1):
        epoch_path = model_dir + "/epoch_" + str(epoch)
        infer(test_reader=test_reader, vocab_tag=vocab_tag, use_cuda=False, model_path=epoch_path)
