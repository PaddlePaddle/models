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
from paddlenlp.transformers.model_bert import *
from paddlenlp.transformers.tokenizer_bert import BertTokenizer

from run_glue import convert_example, TASK_CLASSES

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer), }


def parse_args():
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
        default=None,
        type=str,
        required=True,
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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

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
        "--eager_run", type=eval, default=True, help="Use dygraph mode.")
    parser.add_argument(
        "--use_gpu", type=eval, default=True, help="Whether to use gpu.")
    args = parser.parse_args()
    return args


def do_prdict(args):
    paddle.enable_static() if not args.eager_run else None
    paddle.set_device("gpu" if args.n_gpu else "cpu")

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

    model = model_class.from_pretrained(args.model_name_or_path)

    model.eval()
    for batch in test_data_loader:
        input_ids, segment_ids = batch
        logits = model(input_ids, segment_ids)
        for i, rs in enumerate(paddle.argmax(logits).numpy()):
            print(batch[i], rs)


if __name__ == "__main__":
    args = parse_args()
    do_prdict(args)
