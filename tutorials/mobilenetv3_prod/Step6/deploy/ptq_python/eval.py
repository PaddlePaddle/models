# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import sys
import argparse
import math

sys.path[0] = os.path.join(
    os.path.dirname("__file__"), os.path.pardir, os.path.pardir)

import paddle
import paddle.inference as paddle_infer

from presets import ClassificationPresetEval
import paddlevision


def eval():
    # create predictor
    model_file = os.path.join(FLAGS.model_path, FLAGS.model_filename)
    params_file = os.path.join(FLAGS.model_path, FLAGS.params_filename)
    config = paddle_infer.Config(model_file, params_file)
    if FLAGS.use_gpu:
        config.enable_use_gpu(1000, 0)
    if not FLAGS.ir_optim:
        config.switch_ir_optim(False)

    predictor = paddle_infer.create_predictor(config)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])

    # prepare data
    resize_size, crop_size = (256, 224)
    val_dataset = paddlevision.datasets.ImageFolder(
        os.path.join(FLAGS.data_dir, 'val'),
        ClassificationPresetEval(
            crop_size=crop_size, resize_size=resize_size))

    eval_loader = paddle.io.DataLoader(
        val_dataset, batch_size=FLAGS.batch_size, num_workers=5)

    cost_time = 0.
    total_num = 0.
    correct_1_num = 0
    correct_5_num = 0
    for batch_id, data in enumerate(eval_loader()):
        # set input
        img_np = np.array([tensor.numpy() for tensor in data[0]])
        label_np = np.array([tensor.numpy() for tensor in data[1]])

        input_handle.reshape(img_np.shape)
        input_handle.copy_from_cpu(img_np)

        # run
        t1 = time.time()
        predictor.run()
        t2 = time.time()
        cost_time += (t2 - t1)

        output_data = output_handle.copy_to_cpu()

        # calculate accuracy
        for i in range(len(label_np)):
            label = label_np[i][0]
            result = output_data[i, :]
            index = result.argsort()
            total_num += 1
            if index[-1] == label:
                correct_1_num += 1
            if label in index[-5:]:
                correct_5_num += 1

        if batch_id % 10 == 0:
            acc1 = correct_1_num / total_num
            acc5 = correct_5_num / total_num
            avg_time = cost_time / total_num
            print(
                "batch_id {}, acc1 {:.3f}, acc5 {:.3f}, avg time {:.5f} sec/img".
                format(batch_id, acc1, acc5, avg_time))

    acc1 = correct_1_num / total_num
    acc5 = correct_5_num / total_num
    avg_time = cost_time / total_num
    print("End test: test image {}".format(total_num))
    print("test_acc1 {:.4f}, test_acc5 {:.4f}, avg time {:.5f} sec/img".format(
        acc1, acc5, avg_time))
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path', type=str, default="", help="The inference model path.")
    parser.add_argument(
        '--model_filename',
        type=str,
        default="model.pdmodel",
        help="model filename")
    parser.add_argument(
        '--params_filename',
        type=str,
        default="model.pdiparams",
        help="params filename")
    parser.add_argument(
        '--data_dir',
        type=str,
        default="dataset/ILSVRC2012/",
        help="The ImageNet dataset root dir.")
    parser.add_argument(
        '--batch_size', type=int, default=10, help="Batch size.")
    parser.add_argument(
        '--use_gpu', type=bool, default=False, help=" Whether use gpu or not.")
    parser.add_argument(
        '--ir_optim', type=bool, default=False, help="Enable ir optim.")

    FLAGS = parser.parse_args()

    eval()
