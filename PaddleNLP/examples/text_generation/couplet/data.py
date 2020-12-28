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
from paddlenlp.data import Vocab, Pad
from paddlenlp.data import SamplerHelper
from paddlenlp.datasets import TranslationDataset


def create_train_loader(batch_size=128):
    train_ds = CoupletDataset.get_datasets(["train"])
    vocab, _ = CoupletDataset.get_vocab()
    pad_id = vocab[CoupletDataset.EOS_TOKEN]

    train_batch_sampler = SamplerHelper(train_ds).shuffle().batch(
        batch_size=batch_size)

    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=partial(
            prepare_input, pad_id=pad_id))

    return train_loader, len(vocab), pad_id


def create_infer_loader(batch_size=128):
    test_ds = CoupletDataset.get_datasets(["test"])
    vocab, _ = CoupletDataset.get_vocab()
    pad_id = vocab[CoupletDataset.EOS_TOKEN]
    bos_id = vocab[CoupletDataset.BOS_TOKEN]
    eos_id = vocab[CoupletDataset.EOS_TOKEN]

    test_batch_sampler = SamplerHelper(test_ds).batch(batch_size=batch_size)

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


class CoupletDataset(TranslationDataset):
    URL = "https://paddlenlp.bj.bcebos.com/datasets/couplet.tar.gz"
    SPLITS = {
        'train': TranslationDataset.META_INFO(
            os.path.join("couplet", "train_src.tsv"),
            os.path.join("couplet", "train_tgt.tsv"),
            "ad137385ad5e264ac4a54fe8c95d1583",
            "daf4dd79dbf26040696eee0d645ef5ad"),
        'dev': TranslationDataset.META_INFO(
            os.path.join("couplet", "dev_src.tsv"),
            os.path.join("couplet", "dev_tgt.tsv"),
            "65bf9e72fa8fdf0482751c1fd6b6833c",
            "3bc3b300b19d170923edfa8491352951"),
        'test': TranslationDataset.META_INFO(
            os.path.join("couplet", "test_src.tsv"),
            os.path.join("couplet", "test_tgt.tsv"),
            "f0a7366dfa0acac884b9f4901aac2cc1",
            "56664bff3f2edfd7a751a55a689f90c2")
    }
    VOCAB_INFO = (os.path.join("couplet", "vocab.txt"), os.path.join(
        "couplet", "vocab.txt"), "0bea1445c7c7fb659b856bb07e54a604",
                  "0bea1445c7c7fb659b856bb07e54a604")
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'
    MD5 = '5c0dcde8eec6a517492227041c2e2d54'

    def __init__(self, mode='train', root=None):
        data_select = ('train', 'dev', 'test')
        if mode not in data_select:
            raise TypeError(
                '`train`, `dev` or `test` is supported but `{}` is passed in'.
                format(mode))
        # Download and read data
        self.data = self.get_data(mode=mode, root=root)
        self.vocab, _ = self.get_vocab(root)
        self.transform()

    def transform(self):
        eos_id = self.vocab[self.EOS_TOKEN]
        bos_id = self.vocab[self.BOS_TOKEN]
        self.data = [(
            [bos_id] + self.vocab.to_indices(data[0].split("\x02")) + [eos_id],
            [bos_id] + self.vocab.to_indices(data[1].split("\x02")) + [eos_id])
                     for data in self.data]
