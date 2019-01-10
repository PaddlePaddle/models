import os
import sys
import time
import unittest
import contextlib
import numpy as np

import paddle
import paddle.fluid as fluid

import data_reader


def infer(test_reader, use_cuda, model_path=None):
    """
    inference function
    """
    if model_path is None:
        print(str(model_path) + " cannot be found")
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)

        total_acc = 0.0
        total_count = 0
        for data in test_reader():
            acc = exe.run(inference_program,
                          feed=utils.data2tensor(data, place),
                          fetch_list=fetch_targets,
                          return_numpy=True)
            total_acc += acc[0] * len(data)
            total_count += len(data)

        avg_acc = total_acc / total_count
        print("model_path: %s, avg_acc: %f" % (model_path, avg_acc))


if __name__ == "__main__":
    if __package__ is None:
        from os import sys, path
        sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import utils

    batch_size = 128
    model_path = sys.argv[1]
    test_data_dirname = 'test_data'

    if len(sys.argv) == 3:
        test_data_dirname = sys.argv[2]

    test_reader = data_reader.imdb_data_feed_reader(
        'test_data', batch_size, buf_size=500000)

    models = os.listdir(model_path)
    for i in range(0, len(models)):
        epoch_path = "epoch" + str(i) + ".model"
        epoch_path = os.path.join(model_path, epoch_path)
        infer(test_reader, use_cuda=False, model_path=epoch_path)
