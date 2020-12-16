# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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

import os
import sys
import time
import six
import numpy as np
import math
import argparse
import logging
import paddle.fluid as fluid
import paddle
import time
import reader as reader
from nets import MultiviewSimnet, SimpleEncoderFactory

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def check_version():
    """
     Log error and exit when the installed version of paddlepaddle is
     not satisfied.
     """
    err = "PaddlePaddle version 1.6 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.6.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser("multi-view simnet")
    parser.add_argument("--train_file", type=str, help="Training file")
    parser.add_argument("--valid_file", type=str, help="Validation file")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument(
        "--model_dir",
        type=str,
        default='model_output',
        help="Model output folder")
    parser.add_argument(
        "--query_slots", type=int, default=1, help="Number of query slots")
    parser.add_argument(
        "--title_slots", type=int, default=1, help="Number of title slots")
    parser.add_argument(
        "--query_encoder",
        type=str,
        default="bow",
        help="Encoder module for slot encoding")
    parser.add_argument(
        "--title_encoder",
        type=str,
        default="bow",
        help="Encoder module for slot encoding")
    parser.add_argument(
        "--query_encode_dim",
        type=int,
        default=128,
        help="Dimension of query encoder output")
    parser.add_argument(
        "--title_encode_dim",
        type=int,
        default=128,
        help="Dimension of title encoder output")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Default Dimension of Embedding")
    parser.add_argument(
        "--sparse_feature_dim",
        type=int,
        default=1000001,
        help="Sparse feature hashing space for index processing")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden dim")
    return parser.parse_args()


def start_infer(args, model_path):
    dataset = reader.SyntheticDataset(args.sparse_feature_dim, args.query_slots,
                                      args.title_slots)
    test_reader = fluid.io.batch(
        fluid.io.shuffle(
            dataset.valid(), buf_size=args.batch_size * 100),
        batch_size=args.batch_size)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    with fluid.scope_guard(fluid.Scope()):
        infer_program, feed_target_names, fetch_vars = fluid.io.load_inference_model(
            args.model_dir, exe)
        t0 = time.time()
        step_id = 0
        feeder = fluid.DataFeeder(
            program=infer_program, feed_list=feed_target_names, place=place)
        for batch_id, data in enumerate(test_reader()):
            step_id += 1
            loss_val, correct_val = exe.run(infer_program,
                                            feed=feeder.feed(data),
                                            fetch_list=fetch_vars)
            logger.info("TRAIN --> pass: {} batch_id: {} avg_cost: {}, acc: {}"
                        .format(step_id, batch_id, loss_val,
                                float(correct_val) / args.batch_size))


def main():
    args = parse_args()
    start_infer(args, args.model_dir)


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    check_version()
    main()
