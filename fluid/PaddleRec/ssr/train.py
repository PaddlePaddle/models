#Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import argparse
import logging
import paddle.fluid as fluid
import paddle
import utils
import numpy as np
from nets import SequenceSemanticRetrieval

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("sequence semantic retrieval")
    parser.add_argument(
        "--train_dir", type=str, default='train_data', help="Training file")
    parser.add_argument(
        "--base_lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        '--vocab_path', type=str, default='vocab.txt', help='vocab file')
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        '--parallel', type=int, default=0, help='whether parallel')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether use gpu')
    parser.add_argument(
        '--print_batch', type=int, default=10, help='num of print batch')
    parser.add_argument(
        '--model_dir', type=str, default='model_output', help='model dir')
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="hidden size")
    parser.add_argument(
        "--batch_size", type=int, default=50, help="number of batch")
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="embedding dim")
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    return parser.parse_args()


def get_cards(args):
    return args.num_devices


def train(args):
    use_cuda = True if args.use_cuda else False
    parallel = True if args.parallel else False
    print("use_cuda:", use_cuda, "parallel:", parallel)
    train_reader, vocab_size = utils.construct_train_data(
        args.train_dir, args.vocab_path, args.batch_size * get_cards(args))
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    ssr = SequenceSemanticRetrieval(vocab_size, args.embedding_dim,
                                    args.hidden_size)
    # Train program
    train_input_data, cos_pos, avg_cost, acc = ssr.train()

    # Optimization to minimize lost
    optimizer = fluid.optimizer.Adagrad(learning_rate=args.base_lr)
    optimizer.minimize(avg_cost)

    data_list = [var.name for var in train_input_data]
    feeder = fluid.DataFeeder(feed_list=data_list, place=place)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    if parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda, loss_name=avg_cost.name)
    else:
        train_exe = exe

    total_time = 0.0
    for pass_id in range(args.epochs):
        epoch_idx = pass_id + 1
        print("epoch_%d start" % epoch_idx)
        t0 = time.time()
        i = 0
        for batch_id, data in enumerate(train_reader()):
            i += 1
            loss_val, correct_val = train_exe.run(
                feed=feeder.feed(data), fetch_list=[avg_cost.name, acc.name])
            if i % args.print_batch == 0:
                logger.info(
                    "Train --> pass: {} batch_id: {} avg_cost: {}, acc: {}".
                    format(pass_id, batch_id,
                           np.mean(loss_val),
                           float(np.mean(correct_val)) / args.batch_size))
        t1 = time.time()
        total_time += t1 - t0
        print("epoch:%d num_steps:%d time_cost(s):%f" %
              (epoch_idx, i, total_time / epoch_idx))
        save_dir = "%s/epoch_%d" % (args.model_dir, epoch_idx)
        fluid.io.save_params(executor=exe, dirname=save_dir)
        print("model saved in %s" % save_dir)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
