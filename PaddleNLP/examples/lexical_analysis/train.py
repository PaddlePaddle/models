#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import ast
import math
import argparse

import numpy as np
import paddle

from data import LacDataset
from model import BiGruCrf
from paddlenlp.data import Pad, Tuple, Stack
from paddlenlp.layers.crf import LinearChainCrfLoss, ViterbiDecoder
from paddlenlp.metrics import ChunkEvaluator

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--data_dir", type=str, default=None, help="The folder where the dataset is located.")
parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init model.")
parser.add_argument("--model_save_dir", type=str, default=None, help="The model will be saved in this path.")
parser.add_argument("--epochs", type=int, default=10, help="Corpus iteration num.")
parser.add_argument("--batch_size", type=int, default=300, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=64, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="If set, use GPU for training.")
parser.add_argument("--base_lr", type=float, default=0.001, help="The basic learning rate that affects the entire network.")
parser.add_argument("--emb_dim", type=int, default=128, help="The dimension in which a word is embedded.")
parser.add_argument("--hidden_size", type=int, default=128, help="The number of hidden nodes in the GRU layer.")
args = parser.parse_args()
# yapf: enable


def train(args):
    if args.use_gpu:
        place = paddle.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        paddle.set_device("gpu")
    else:
        place = paddle.CPUPlace()
        paddle.set_device("cpu")

    # create dataset.
    train_dataset = LacDataset(args.data_dir, mode='train')
    test_dataset = LacDataset(args.data_dir, mode='test')

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=0),  # word_ids
        Stack(),  # length
        Pad(axis=0, pad_val=0),  # label_ids
    ): fn(samples)

    # Create sampler for dataloader
    train_sampler = paddle.io.DistributedBatchSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)
    train_loader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        places=place,
        return_list=True,
        collate_fn=batchify_fn)

    test_sampler = paddle.io.BatchSampler(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True)
    test_loader = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_sampler=test_sampler,
        places=place,
        return_list=True,
        collate_fn=batchify_fn)

    # Define the model netword and its loss
    network = BiGruCrf(args.emb_dim, args.hidden_size, train_dataset.vocab_size,
                       train_dataset.num_labels)
    model = paddle.Model(network)

    # Prepare optimizer, loss and metric evaluator
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.base_lr, parameters=model.parameters())
    crf_loss = LinearChainCrfLoss(network.crf.transitions)
    chunk_evaluator = ChunkEvaluator(
        int(math.ceil((train_dataset.num_labels + 1) / 2.0)),
        "IOB")  # + 1 for START and STOP
    model.prepare(optimizer, crf_loss, chunk_evaluator)
    if args.init_checkpoint:
        model.load(args.init_checkpoint)

    # Start training
    callback = paddle.callbacks.ProgBarLogger(log_freq=10, verbose=3)
    model.fit(train_data=train_loader,
              eval_data=test_loader,
              batch_size=args.batch_size,
              epochs=args.epochs,
              eval_freq=1,
              log_freq=10,
              save_dir=args.model_save_dir,
              save_freq=1,
              drop_last=True,
              shuffle=True,
              callbacks=callback)


if __name__ == "__main__":
    print(args)
    train(args)
