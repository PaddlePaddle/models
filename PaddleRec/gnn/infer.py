#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import argparse
import logging
import numpy as np
import os
import paddle
import paddle.fluid as fluid
import reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle DIN example")
    parser.add_argument(
        '--model_path', type=str, default='./saved_model/', help="path of model parameters")
    parser.add_argument(
        '--test_path', type=str, default='./data/diginetica/test.txt', help='dir of test file')
    parser.add_argument(
        '--use_cuda', type=int, default=1, help='whether to use gpu')
    parser.add_argument(
        '--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument(
        '--start_index', type=int, default='0', help='start index')
    parser.add_argument(
        '--last_index', type=int, default='10', help='end index')
    return parser.parse_args()


def infer(epoch_num):
    args = parse_args()
    batch_size = args.batch_size
    test_data = reader.Data(args.test_path, False)
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model_path = args.model_path + "epoch_" + str(epoch_num)
    try:
        [infer_program, feed_names, fetch_targets] = fluid.io.load_inference_model(
            model_path, exe)
        feeder = fluid.DataFeeder(
            feed_list=feed_names, place=place, program=infer_program)

        loss_sum = 0.0
        acc_sum = 0.0
        count = 0
        for data in test_data.reader(batch_size, batch_size, False):
            res = exe.run(infer_program,
                          feed=feeder.feed(data),
                          fetch_list=fetch_targets)
            loss_sum += res[0]
            acc_sum += res[1]
            count += 1
        logger.info("TEST --> loss: %.4lf, Recall@20: %.4lf" %
                    (loss_sum / count, acc_sum / count))
    except ValueError as e:
        logger.info("TEST --> error: there is no model in " + model_path)


if __name__ == "__main__":
    args = parse_args()
    for index in range(args.start_index, args.last_index + 1):
        infer(index)
