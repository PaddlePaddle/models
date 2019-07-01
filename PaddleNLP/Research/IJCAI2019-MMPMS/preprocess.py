#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################

import os
import argparse

from mmpms.inputters.dataset import PostResponseDataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument(
    "--embed_file", type=str, default="./data/glove.840B.300d.txt")
parser.add_argument("--max_vocab_size", type=int, default=30000)
parser.add_argument("--min_len", type=int, default=3)
parser.add_argument("--max_len", type=int, default=30)
args = parser.parse_args()

vocab_file = os.path.join(args.data_dir, "vocab.json")
raw_train_file = os.path.join(args.data_dir, "dial.train")
raw_valid_file = os.path.join(args.data_dir, "dial.valid")
raw_test_file = os.path.join(args.data_dir, "dial.test")
train_file = raw_train_file + ".pkl"
valid_file = raw_valid_file + ".pkl"
test_file = raw_test_file + ".pkl"

dataset = PostResponseDataset(
    max_vocab_size=args.max_vocab_size,
    min_len=args.min_len,
    max_len=args.max_len,
    embed_file=args.embed_file)

# Build vocabulary
dataset.build_vocab(raw_train_file)
dataset.save_vocab(vocab_file)

# Build examples
valid_examples = dataset.build_examples(raw_valid_file)
dataset.save_examples(valid_examples, valid_file)

test_examples = dataset.build_examples(raw_test_file)
dataset.save_examples(test_examples, test_file)

train_examples = dataset.build_examples(raw_train_file)
dataset.save_examples(train_examples, train_file)
