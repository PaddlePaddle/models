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

import argparse
import collections

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, Dataset
import sentencepiece as spm
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default='~/',
        help="vocab file used to tokenize text")
    parser.add_argument(
        "--vocab_model_file",
        type=str,
        default='gpt2.model',
        help="vocab model file used to tokenize text")
    parser.add_argument(
        "--max_encoder_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after SentencePiece tokenization."
    )
    parser.add_argument(
        "--init_from_check_point",
        type=str,
        default=None,
        help="The path of checkpoint to be loaded.")
    args = parser.parse_args()
    return args


def create_tokenizer(model_file):
    return spm.SentencePieceProcessor(model_file=model_file)


class ImdbDataset(Dataset):
    def __init__(self, input_file, tokenizer, max_encoder_length=512,
                 pad_val=0):
        self.samples = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                line = line.rstrip("\n")
                sample = line.split(",", 1)
                label = np.array(int(sample[0])).astype('int32')
                # Add [CLS] (65) and [SEP] (66) special tokens.
                input_ids = [65]
                input_ids.extend(
                    tokenizer.tokenize(sample[1])[:max_encoder_length - 2])
                input_ids.append(66)
                input_len = len(input_ids)
                if input_len < max_encoder_length:
                    input_ids.extend([pad_val] *
                                     (max_encoder_length - input_len))
                input_ids = np.array(input_ids).astype('int32')
                self.samples.append([input_ids, label])

    def __getitem__(self, index):
        # [input_ids, label]
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def create_dataloader(data_dir, batch_size, vocab_model_file,
                      max_encoder_length):
    def _create_dataloader(mode, tokenizer, max_encoder_length, pad_val):
        input_file = os.path.join(data_dir, mode + ".csv")
        dataset = ImdbDataset(input_file, tokenizer, max_encoder_length,
                              pad_val)
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=(mode == "train"))
        data_loader = paddle.io.DataLoader(
            dataset=dataset, batch_sampler=batch_sampler, return_list=True)
        return data_loader

    tokenizer = create_tokenizer(vocab_model_file)
    train_data_loader = _create_dataloader("train", tokenizer,
                                           max_encoder_length, 0)
    test_data_loader = _create_dataloader("test", tokenizer, max_encoder_length,
                                          0)
    return train_data_loader, test_data_loader


def do_train(args):
    # get dataloader
    train_data_loader, test_data_loader = \
            create_dataloader(args.data_dir, args.batch_size, args.vocab_model_file, args.max_encoder_length)
    # define model

    # define metric

    # define optimizer

    # train

    # eval


if __name__ == "__main__":
    args = parse_args()
    if args.n_gpu > 1:
        dist.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
