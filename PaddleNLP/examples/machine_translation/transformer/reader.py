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

import glob
import sys
import os
import io
import itertools
from functools import partial

import numpy as np
from paddle.io import BatchSampler, DataLoader, Dataset
from paddlenlp.data import Pad
from paddlenlp.datasets import WMT14
from paddlenlp.data.sampler import SamplerHelper


def min_max_filer(data, max_len, min_len=0):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


def create_data_loader(args):
    root = None if args.root == "None" else args.root
    transform_func = WMT14.get_default_transform_func(root=root)
    datasets = WMT14(transform_func=transform_func).get_datasets(
        ["train", "dev"])

    def token_size_fn(current_idx, current_batch_size, tokens_sofar,
                      data_source):
        return tokens_sofar + (max(
            len(data_source[current_idx][0]), len(data_source[current_idx][1]))
                               + 1) * current_batch_size

    data_loaders = [(None, None)] * 2
    for i, dataset in enumerate(datasets):
        sampler = SamplerHelper(
            dataset.filter(partial(
                min_max_filer, max_len=args.max_length))).shuffle()

        if args.sort_type == SortType.GLOBAL or args.sort_type == SortType.POOL:
            # else for SortType.GLOBAL
            buffer_size = args.pool_size if args.sort_type == SortType.POOL else -1
            trg_key = (lambda x, data_source: len(data_source[x][1]) + 1)
            src_key = (lambda x, data_source: len(data_source[x][0]) + 1)
            # Sort twice
            sampler = sampler.sort(
                key=trg_key, buffer_size=buffer_size).sort(
                    key=src_key, buffer_size=buffer_size)

        batch_sampler = sampler.batch(
            batch_size=args.batch_size,
            drop_last=False,
            batch_size_fn=token_size_fn).shard()

        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                prepare_train_input,
                bos_idx=args.bos_idx,
                eos_idx=args.eos_idx,
                pad_idx=args.bos_idx),
            num_workers=0,
            return_list=True)
        data_loaders[i] = (data_loader, batch_sampler.__len__)
    return data_loaders


def create_infer_loader(args):
    root = None if args.root == "None" else args.root
    (src_vocab, tgt_vocab) = WMT14.get_vocab(root=root)
    transform_func = WMT14.get_default_transform_func(root=root)
    dataset = WMT14(transform_func=transform_func).get_datasets(
        ["test"]).filter(partial(
            min_max_filer, max_len=args.max_length))
    batch_sampler = SamplerHelper(dataset).shuffle().batch(
        batch_size=args.batch_size, drop_last=False).shard()

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            prepare_infer_input,
            bos_idx=args.bos_idx,
            eos_idx=args.eos_idx,
            pad_idx=args.bos_idx),
        num_workers=0,
        return_list=True)
    data_loaders = (data_loader, batch_sampler.__len__)
    return data_loaders, tgt_vocab


def prepare_train_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by training into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])
    trg_word = word_pad([[bos_idx] + inst[1] for inst in insts])
    lbl_word = np.expand_dims(
        word_pad([inst[1] + [eos_idx] for inst in insts]), axis=2)

    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs


def prepare_infer_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])

    return [src_word, ]


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"
