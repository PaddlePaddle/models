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
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from paddlenlp.transformers import BertWithBigBird, create_bigbird_attention_mask_list
from paddlenlp.utils.log import logger
import sentencepiece as spm
import os
import random


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
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="The number of BigBird Encoder layers")
    parser.add_argument(
        "--attention_type", type=str, default="bigbird_simulated", help="")
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=12,
        help="number of heads when compute attention")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="")
    parser.add_argument("--hidden_size", type=int, default=768, help="")
    parser.add_argument("--num_labels", type=int, default=2, help="")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=1000,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--decay_steps",
        default=100,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate used to train.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='chekpoints/',
        help="Directory to save model checkpoint")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epoches for training.")

    args = parser.parse_args()
    return args


def create_tokenizer(model_file):
    return spm.SentencePieceProcessor(model_file=model_file)


class ImdbDataset(Dataset):
    def __init__(self, input_file, tokenizer, max_encoder_length=512,
                 pad_val=0):
        self.samples = []
        if input_file:
            with open(input_file, "r") as f:
                for line in f.readlines():
                    line = line.rstrip("\n")
                    sample = line.split(",", 1)
                    label = np.array(int(sample[0])).astype('int64')
                    # Add [CLS] (65) and [SEP] (66) special tokens.
                    input_ids = [65]
                    input_ids.extend(
                        tokenizer.tokenize(sample[1])[:max_encoder_length - 2])
                    input_ids.append(66)
                    input_len = len(input_ids)
                    if input_len < max_encoder_length:
                        input_ids.extend([pad_val] *
                                         (max_encoder_length - input_len))
                    input_ids = np.array(input_ids).astype('int64')
                    self.samples.append([input_ids, label])

    def split(self, rate):
        num_samples = len(self.samples)
        num_save = int(num_samples * rate)
        random.shuffle(self.samples)
        split_dataset = ImdbDataset(None, None)
        split_dataset.samples = self.samples[num_save:]
        self.samples = self.samples[:num_save]
        return split_dataset

    def __getitem__(self, index):
        # [input_ids, label]
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class ClassifierModel(nn.Layer):
    def __init__(self, num_labels, **kwargv):
        super(ClassifierModel, self).__init__()
        self.bigbird_model = BertWithBigBird(**kwargv)
        self.linear = nn.Linear(kwargv['hidden_size'], num_labels)
        self.dropout = nn.Dropout(
            kwargv['hidden_dropout_prob'], mode="upscale_in_train")
        self.kwargv = kwargv

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        attention_mask_list, rand_mask_idx_list = \
            create_bigbird_attention_mask_list(
                self.kwargv['num_layers'],
                seq_len, seq_len,
                self.kwargv['nhead'],
                self.kwargv['block_size'],
                self.kwargv['window_size'],
                self.kwargv['num_global_blocks'],
                self.kwargv['num_rand_blocks'],
                self.kwargv['seed'],)
        _, pooled_output = self.bigbird_model(
            input_ids,
            None,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)
        output = self.dropout(pooled_output)
        output = self.linear(output)
        return output


def create_dataloader(data_dir, batch_size, max_encoder_length, tokenizer):
    def _create_dataloader(mode,
                           tokenizer,
                           max_encoder_length,
                           pad_val,
                           split_dev=True):
        input_file = os.path.join(data_dir, mode + ".csv")
        dataset = ImdbDataset(input_file, tokenizer, max_encoder_length,
                              pad_val)
        if split_dev:
            split_dataset = dataset.split(0.2)
            split_batch_sampler = paddle.io.BatchSampler(
                split_dataset, batch_size=batch_size)
            split_data_loader = paddle.io.DataLoader(
                dataset=split_dataset,
                batch_sampler=split_batch_sampler,
                return_list=True)
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=(mode == "train"))
        data_loader = paddle.io.DataLoader(
            dataset=dataset, batch_sampler=batch_sampler, return_list=True)
        if split_dev:
            return data_loader, split_data_loader
        return data_loader

    train_data_loader, dev_data_loader = _create_dataloader(
        "train", tokenizer, max_encoder_length, 0)
    test_data_loader = _create_dataloader("test", tokenizer, max_encoder_length,
                                          0, False)
    return train_data_loader, dev_data_loader, test_data_loader


def get_config(args, vocab_size):

    bertConfig = {
        "num_layers": args.num_layers,
        "vocab_size": vocab_size,
        "nhead": args.num_attention_heads,
        "attn_dropout": args.attn_dropout,
        "dim_feedforward": 3072,
        "activation": "gelu",
        "normalize_before": False,
        "attention_type": args.attention_type,
        "block_size": 64,
        "window_size": 3,
        "num_global_blocks": 1,
        "num_rand_blocks": 2,
        "seed": None,
        "pad_token_id": 0,
        "hidden_size": args.hidden_size,
        "hidden_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2
    }
    return bertConfig


def do_train(args):
    # get dataloader
    tokenizer = create_tokenizer(args.vocab_model_file)
    vocab_size = tokenizer.vocab_size()
    train_data_loader, dev_data_loader, test_data_loader = \
            create_dataloader(args.data_dir, args.batch_size, args.max_encoder_length, tokenizer)
    # define model
    bertConfig = get_config(args, vocab_size)
    model = ClassifierModel(args.num_labels, **bertConfig)

    # define metric
    # criterion = nn.CrossEntropyLoss()
    log_softmax = nn.LogSoftmax()
    softmax = nn.Softmax()
    criterion = nn.NLLLoss()
    metric = paddle.metric.Accuracy()

    # define optimizer
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        args.lr,
        lambda current_step, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.decay_steps if args.decay_steps > 0 else
        (len(train_data_loader) * args.num_train_epochs): float(
            current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps else max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps))),
        last_epoch=0)

    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=lr_scheduler,
        weight_decay=args.weight_decay)

    global_steps = 0
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_data_loader):
            global_steps += 1
            (input_ids, labels) = batch
            output = model(input_ids)
            log_prob = log_softmax(output)
            loss = criterion(log_prob, labels)
            prob = softmax(output)
            metric.update(prob, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
            # save model
        # print loss
        logger.info("global step %d, epoch: %d, loss: %f, acc %f" %
                    (global_steps, epoch, loss, metric.accumulate()))


if __name__ == "__main__":
    args = parse_args()
    if args.n_gpu > 1:
        dist.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
