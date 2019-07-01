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

from __future__ import absolute_import

import os
import argparse
from datetime import datetime

from mmpms.utils.args import str2bool
from mmpms.utils.args import parse_args
from mmpms.utils.logging import getLogger
from mmpms.utils.misc import tensor2list

from mmpms.inputters.dataset import PostResponseDataset
from mmpms.inputters.dataloader import DataLoader
from mmpms.models.mmpms import MMPMS
from mmpms.modules.generator import BeamSearch
from mmpms.engine import Engine
from mmpms.engine import evaluate
from mmpms.engine import infer

parser = argparse.ArgumentParser()
parser.add_argument("--args_file", type=str, default=None)
parser.add_argument("--use_gpu", type=str2bool, default=True)
parser.add_argument("--model_dir", type=str, default=None)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--infer", action="store_true")

# Data
data_arg = parser.add_argument_group("Data")
data_arg.add_argument("--data_dir", type=str, default="./data/")
data_arg.add_argument("--vocab_file", type=str, default=None)
data_arg.add_argument("--train_file", type=str, default=None)
data_arg.add_argument("--valid_file", type=str, default=None)
data_arg.add_argument("--test_file", type=str, default=None)
parser.add_argument(
    "--embed_file", type=str, default="./data/glove.840B.300d.txt")

data_arg.add_argument("--max_vocab_size", type=int, default=30000)
data_arg.add_argument("--min_len", type=int, default=3)
data_arg.add_argument("--max_len", type=int, default=30)

# Model
model_arg = parser.add_argument_group("Model")
model_arg.add_argument("--embed_dim", type=int, default=300)
model_arg.add_argument("--hidden_dim", type=int, default=1024)
model_arg.add_argument("--num_mappings", type=int, default=20)
model_arg.add_argument("--tau", type=float, default=0.67)
model_arg.add_argument("--num_layers", type=int, default=1)
model_arg.add_argument("--bidirectional", type=str2bool, default=True)
model_arg.add_argument(
    "--attn_mode",
    type=str,
    default='mlp',
    choices=['none', 'mlp', 'dot', 'general'])
model_arg.add_argument(
    "--use_pretrained_embedding", type=str2bool, default=True)
model_arg.add_argument("--embed_init_scale", type=float, default=0.03)
model_arg.add_argument("--dropout", type=float, default=0.3)

# Training
train_arg = parser.add_argument_group("Train")
train_arg.add_argument("--save_dir", type=str, default="./output/")
train_arg.add_argument("--num_epochs", type=int, default=10)
train_arg.add_argument("--shuffle", type=str2bool, default=True)
train_arg.add_argument("--log_steps", type=int, default=100)
train_arg.add_argument("--valid_steps", type=int, default=500)
train_arg.add_argument("--batch_size", type=int, default=128)

# Optimization
optim_arg = parser.add_argument_group("Optim")
optim_arg.add_argument("--optimizer", type=str, default="Adam")
optim_arg.add_argument("--lr", type=float, default=0.0002)
optim_arg.add_argument("--grad_clip", type=float, default=5.0)

# Inference
infer_arg = parser.add_argument_group("Inference")
infer_arg.add_argument("--beam_size", type=int, default=10)
infer_arg.add_argument("--min_infer_len", type=int, default=3)
infer_arg.add_argument("--max_infer_len", type=int, default=30)
infer_arg.add_argument("--length_average", type=str2bool, default=False)
infer_arg.add_argument("--ignore_unk", type=str2bool, default=True)
infer_arg.add_argument("--ignore_repeat", type=str2bool, default=True)
infer_arg.add_argument("--infer_batch_size", type=int, default=64)
infer_arg.add_argument("--result_file", type=str, default="./infer.result")


def main():
    args = parse_args(parser)

    if args.args_file:
        args.load(args.args_file)
        print("Loaded args from '{}'".format(args.args_file))

    args.Data.vocab_file = args.Data.vocab_file or os.path.join(
        args.Data.data_dir, "vocab.json")
    args.Data.train_file = args.Data.train_file or os.path.join(
        args.Data.data_dir, "dial.train.pkl")
    args.Data.valid_file = args.Data.valid_file or os.path.join(
        args.Data.data_dir, "dial.valid.pkl")
    args.Data.test_file = args.Data.test_file or os.path.join(
        args.Data.data_dir, "dial.test.pkl")

    print("Args:")
    print(args)
    print()

    # Dataset Definition
    dataset = PostResponseDataset(
        max_vocab_size=args.max_vocab_size,
        min_len=args.min_len,
        max_len=args.max_len,
        embed_file=args.embed_file)
    dataset.load_vocab(args.vocab_file)

    # Generator Definition
    generator = BeamSearch(
        vocab_size=dataset.vocab.size(),
        beam_size=args.beam_size,
        start_id=dataset.vocab.bos_id,
        end_id=dataset.vocab.eos_id,
        unk_id=dataset.vocab.unk_id,
        min_length=args.min_infer_len,
        max_length=args.max_infer_len,
        length_average=args.length_average,
        ignore_unk=args.ignore_unk,
        ignore_repeat=args.ignore_repeat)

    # Model Definition
    model = MMPMS(
        vocab=dataset.vocab,
        generator=generator,
        hparams=args.Model,
        optim_hparams=args.Optim,
        use_gpu=args.use_gpu)

    infer_parse_dict = {
        "post": lambda T: dataset.denumericalize(tensor2list(T)),
        "response": lambda T: dataset.denumericalize(tensor2list(T)),
        "preds": lambda T: dataset.denumericalize(tensor2list(T)),
    }

    if args.infer:
        if args.model_dir is not None:
            model.load(args.model_dir)
            print("Loaded model checkpoint from '{}'".format(args.model_dir))

        infer_data = dataset.load_examples(args.test_file)
        infer_loader = DataLoader(
            data=infer_data,
            batch_size=args.infer_batch_size,
            shuffle=False,
            use_gpu=args.use_gpu)

        print("Inference starts ...")
        infer_results = infer(
            model, infer_loader, infer_parse_dict, save_file=args.result_file)

    elif args.eval:
        if args.model_dir is not None:
            model.load(args.model_dir)
            print("Loaded model checkpoint from '{}'".format(args.model_dir))

        eval_data = dataset.load_examples(args.test_file)
        eval_loader = DataLoader(
            data=eval_data,
            batch_size=args.batch_size,
            shuffle=False,
            use_gpu=args.use_gpu)

        print("Evaluation starts ...")
        eval_metrics_tracker = evaluate(model, eval_loader)
        print("   ".join("{}-{}".format(name.upper(), value.avg)
                         for name, value in eval_metrics_tracker.items()))

    else:
        valid_data = dataset.load_examples(args.valid_file)
        valid_loader = DataLoader(
            data=valid_data,
            batch_size=args.batch_size,
            shuffle=False,
            use_gpu=args.use_gpu)

        train_data = dataset.load_examples(args.train_file)
        train_loader = DataLoader(
            data=train_data,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            use_gpu=args.use_gpu)

        # Save Directory Definition
        date_str, time_str = datetime.now().strftime("%Y%m%d-%H%M%S").split("-")
        result_str = "{}-{}".format(model.__class__.__name__, time_str)
        args.save_dir = os.path.join(args.save_dir, date_str, result_str)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # Logger Definition
        logger = getLogger(
            os.path.join(args.save_dir, "train.log"), name="mmpms")

        # Save args
        args_file = os.path.join(args.save_dir, "args.json")
        args.save(args_file)
        logger.info("Saved args to '{}'".format(args_file))

        # Executor Definition
        exe = Engine(
            model=model,
            save_dir=args.save_dir,
            log_steps=args.log_steps,
            valid_steps=args.valid_steps,
            logger=logger)

        if args.model_dir is not None:
            exe.load(args.model_dir)

        # Train
        logger.info("Training starts ...")
        exe.evaluate(valid_loader, is_save=False)
        for epoch in range(args.num_epochs):
            exe.train_epoch(train_iter=train_loader, valid_iter=valid_loader)
        logger.info("Training done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
