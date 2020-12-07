# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import itertools
import os
import sys
import hashlib
import random
import time
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader

from paddlenlp.datasets.dataset import *
from paddlenlp.datasets.glue import *
from paddlenlp.data import *
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import ElectraForSequenceClassification, ElectraTokenizer

from run_glue import convert_example, TASK_CLASSES

MODEL_CLASSES = {
    "electra": (ElectraForSequenceClassification, ElectraTokenizer),
}


def do_prdict(args):
    paddle.set_device("gpu" if args.use_gpu else "cpu")

    args.task_name = args.task_name.lower()
    dataset_class, _ = TASK_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    test_dataset = dataset_class.get_datasets(["test"])
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=test_dataset.get_labels(),
        max_seq_length=args.max_seq_length,
        is_test=True)
    test_dataset = test_dataset.apply(trans_func, lazy=True)
    test_batch_sampler = paddle.io.BatchSampler(
        test_dataset, batch_size=args.batch_size, shuffle=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # segment
        Stack(),  # length
    ): fn(samples)[:2]
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_sampler=test_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    # for debug
    model = model_class.from_pretrained(args.model_name_or_path)
    return_dict = model.return_dict

    model.eval()

    for batch in test_data_loader:
        input_ids, segment_ids = batch
        model_output = model(input_ids=input_ids, token_type_ids=segment_ids)
        if not return_dict:
            logits = model_output[0]
        else:
            logits = model_output.logits
        #print("logits.shape is : %s" % logits.shape)
        for i, rs in enumerate(paddle.argmax(logits, -1).numpy()):
            print("data : %s, predict : %s" % (input_ids[i], rs))


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(TASK_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default="electra",
        type=str,
        required=False,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for prediction.", )
    parser.add_argument(
        "--use_gpu", type=eval, default=True, help="Whether to use gpu.")
    args, unparsed = parser.parse_known_args()
    print_arguments(args)
    do_prdict(args)
