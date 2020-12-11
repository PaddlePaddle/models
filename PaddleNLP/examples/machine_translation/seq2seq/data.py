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

import io
import os

from functools import partial
import numpy as np

import paddle
from paddle.utils.download import get_path_from_url
from paddlenlp.data import Vocab, Pad
from paddlenlp.data import SamplerHelper
from paddlenlp.utils.env import DATA_HOME

from paddlenlp.datasets import IWSLT15


def create_train_loader(batch_size=128):
    train_ds, dev_ds = IWSLT15.get_datasets(["train", "dev"])

    vocab = IWSLT15.build_vocab()
    pad_id = vocab[IWSLT15.eos_token]

    train_batch_sampler = SamplerHelper(train_ds).shuffle().sort(
    ).batch(  # TODO
        batch_size=batch_size, drop_last=True).shard()

    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=partial(
            prepare_input, pad_id=pad_id))

    dev_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=partial(
            prepare_input, pad_id=pad_id))

    return train_loader, dev_loader, len(vocab), pad_id


def create_infer_loader(batch_size=128):
    test_ds = IWSLT15.get_datasets(["test"])

    vocab = IWSLT15.build_vocab()
    pad_id = vocab[IWSLT15.eos_token]
    bos_id = vocab[IWSLT15.bos_token]
    eos_id = vocab[IWSLT15.eos_token]

    test_batch_sampler = SamplerHelper(test_ds).batch(
        batch_size=batch_size, drop_last=True).shard()

    test_loader = paddle.io.DataLoader(
        test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=partial(
            prepare_input, pad_id=pad_id))
    return test_loader, len(vocab), pad_id, bos_id, eos_id


def prepare_input(insts, pad_id):
    src, src_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[0] for inst in insts])
    tgt, tgt_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[1] for inst in insts])
    tgt_mask = (tgt[:, :-1] != pad_id).astype(paddle.get_default_dtype())
    return src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis], tgt_mask
