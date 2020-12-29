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
import logging
import os
import random
import time
import h5py
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, Dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import BertForPretraining, BertModel, BertPretrainingCriterion
from paddlenlp.transformers import BertTokenizer
from paddlenlp.utils.log import logger

MODEL_CLASSES = {"bert": (BertForPretraining, BertTokenizer), }


def create_pretrained_dataset(args):
    split = args.split


def set_seed(args):
    random.seed(args.seed + paddle.distributed.get_rank())
    np.random.seed(args.seed + paddle.distributed.get_rank())
    paddle.seed(args.seed + paddle.distributed.get_rank())


def parse_args():
    parser = argparse.ArgumentParser()
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
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.", )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--max_predictions_per_seq",
        default=80,
        type=int,
        help="The maximum total of masked tokens in input sequence")

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
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--eager_run", type=eval, default=True, help="Use dygraph mode.")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    args = parser.parse_args()
    return args


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def do_train(args):
    paddle.enable_static() if not args.eager_run else None
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    worker_init = WorkerInitObj(args.seed + paddle.distributed.get_rank())

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    model = BertForPretraining(
        BertModel(**model_class.pretrained_init_configuration[
            args.model_name_or_path]))
    criterion = BertPretrainingCriterion(
        getattr(model, BertForPretraining.base_model_prefix).config[
            "vocab_size"])
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # If use defalut last_epoch, lr of the first iteration is 0.
    # Use `last_epoch = 0` to be consistent with nv bert.
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        args.learning_rate,
        lambda current_step, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps if args.max_steps > 0 else
        (len(train_data_loader) * args.num_train_epochs): float(
            current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps else max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps))),
        last_epoch=0)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    pool = ThreadPoolExecutor(1)
    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if os.path.isfile(os.path.join(args.input_dir, f)) and "training" in
            f
        ]
        files.sort()
        num_files = len(files)
        random.Random(args.seed + epoch).shuffle(files)
        f_start_id = 0

        shared_file_list = {}

        if paddle.distributed.get_world_size() > num_files:
            remainder = paddle.distributed.get_world_size() % num_files
            data_file = files[(
                f_start_id * paddle.distributed.get_world_size() +
                paddle.distributed.get_rank() + remainder * f_start_id) %
                              num_files]
        else:
            data_file = files[(f_start_id * paddle.distributed.get_world_size()
                               + paddle.distributed.get_rank()) % num_files]

        previous_file = data_file

        train_data_loader, _ = create_pretraining_dataset(
            data_file, args.max_predictions_per_seq, shared_file_list, args,
            worker_init)

        for f_id in range(f_start_id + 1, len(files)):
            if paddle.distributed.get_world_size() > num_files:
                data_file = files[(
                    f_id * paddle.distributed.get_world_size() +
                    paddle.distributed.get_rank() + remainder * f_id) %
                                  num_files]
            else:
                data_file = files[(f_id * paddle.distributed.get_world_size() +
                                   paddle.distributed.get_rank()) % num_files]

            previous_file = data_file
            dataset_future = pool.submit(create_pretraining_dataset, data_file,
                                         args.max_predictions_per_seq,
                                         shared_file_list, args, worker_init)
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                (input_ids, segment_ids, input_mask, masked_lm_positions,
                 masked_lm_labels, next_sentence_labels,
                 masked_lm_scale) = batch
                prediction_scores, seq_relationship_score = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    masked_positions=masked_lm_positions)
                loss = criterion(prediction_scores, seq_relationship_score,
                                 masked_lm_labels, next_sentence_labels,
                                 masked_lm_scale)
                if global_step % args.logging_steps == 0:
                    if (not args.n_gpu > 1
                        ) or paddle.distributed.get_rank() == 0:
                        logger.info(
                            "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                            % (global_step, epoch, step, loss,
                               args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_gradients()
                if global_step % args.save_steps == 0:
                    if (not args.n_gpu > 1
                        ) or paddle.distributed.get_rank() == 0:
                        output_dir = os.path.join(args.output_dir,
                                                  "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        paddle.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "model_state.pdopt"))
                if global_step >= args.max_steps:
                    del train_data_loader
                    return

            del train_data_loader
            train_data_loader, data_file = dataset_future.result(timeout=None)


if __name__ == "__main__":
    args = parse_args()
    main(args)
