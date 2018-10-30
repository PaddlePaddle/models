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
import argparse
import logging
import paddle.fluid as fluid
import paddle
import reader as reader
from nets import SequenceSemanticRetrieval

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser("sequence semantic retrieval")
    parser.add_argument("--train_file", type=str, help="Training file")
    parser.add_argument("--valid_file", type=str, help="Validation file")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default='model_output',
        help="Model output folder")
    parser.add_argument(
        "--sequence_encode_dim",
        type=int,
        default=128,
        help="Dimension of sequence encoder output")
    parser.add_argument(
        "--matching_dim",
        type=int,
        default=128,
        help="Dimension of hidden layer")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Default Dimension of Embedding")
    return parser.parse_args()

def start_train(args):
    y_vocab = reader.YoochooseVocab()
    y_vocab.load([args.train_file])

    logger.info("Load yoochoose vocabulary size: {}".format(len(y_vocab.get_vocab())))
    y_data = reader.YoochooseDataset(y_vocab)
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            y_data.train([args.train_file]), buf_size=args.batch_size * 100),
        batch_size=args.batch_size)
    place = fluid.CPUPlace()
    ssr = SequenceSemanticRetrieval(
        len(y_vocab.get_vocab()), args.embedding_dim, args.matching_dim
    )
    input_data, user_rep, item_rep, avg_cost, acc = ssr.train()
    optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
    optimizer.minimize(avg_cost)
    startup_program = fluid.default_startup_program()
    loop_program = fluid.default_main_program()
    data_list = [var.name for var in input_data]
    feeder = fluid.DataFeeder(feed_list=data_list, place=place)
    exe = fluid.Executor(place)
    exe.run(startup_program)

    for pass_id in range(args.epochs):
        for batch_id, data in enumerate(train_reader()):
            loss_val, correct_val = exe.run(loop_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, acc])
            logger.info("Train --> pass: {} batch_id: {} avg_cost: {}, acc: {}".
                        format(pass_id, batch_id, loss_val, 
                               float(correct_val) / args.batch_size))
        fluid.io.save_inference_model(args.model_output_dir, 
                                      [var.name for val in input_data],
                                      [user_rep, item_rep, avg_cost, acc], exe)

def main():
    args = parse_args()
    start_train(args)

if __name__ == "__main__":
    main()

