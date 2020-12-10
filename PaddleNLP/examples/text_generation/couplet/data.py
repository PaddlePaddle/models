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

DATA_HOME = "/root/.paddlenlp/datasets"


def create_train_loader(batch_size=128):
    train_ds, dev_ds = CoupletDataset.get_datasets(["train", "dev"])

    vocab = CoupletDataset.build_vocab()
    pad_id = vocab[CoupletDataset.eos_token]

    key = (lambda x, data_source: len(data_source[x][0]))

    train_batch_sampler = SamplerHelper(train_ds).shuffle().sort(
        key=key, buffer_size=batch_size * 20).batch(
            batch_size=batch_size, drop_last=True).shard()

    dev_batch_sampler = SamplerHelper(dev_ds).shuffle().sort(
        key=key, buffer_size=batch_size * 20).batch(
            batch_size=batch_size, drop_last=True).shard()

    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=partial(
            prepare_train_input, pad_id=pad_id))

    dev_loader = paddle.io.DataLoader(
        dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=partial(
            prepare_train_input, pad_id=pad_id))

    return train_loader, dev_loader, len(vocab), pad_id


def create_infer_loader(batch_size=128):
    test_ds = CoupletDataset.get_datasets(["test"])

    vocab = CoupletDataset.build_vocab()
    pad_id = vocab[CoupletDataset.eos_token]
    bos_id = vocab[CoupletDataset.bos_token]
    eos_id = vocab[CoupletDataset.eos_token]

    test_batch_sampler = SamplerHelper(test_ds).batch(
        batch_size=batch_size, drop_last=True).shard()

    test_loader = paddle.io.DataLoader(
        test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=partial(
            prepare_train_input, pad_id=pad_id))
    return test_loader, len(vocab), pad_id, bos_id, eos_id


def prepare_train_input(insts, pad_id):
    # add eos, pad_id
    src, src_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[0] for inst in insts])
    tgt, tgt_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[1] for inst in insts])
    tgt_mask = (tgt[:, :-1] != pad_id).astype(paddle.get_default_dtype())
    return src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis], tgt_mask


class CoupletDataset(paddle.io.Dataset):
    URL = "https://bj.bcebos.com/paddlehub-dataset/couplet.tar.gz"
    dataset_dirname = "couplet"
    vocab_filename = "vocab.txt"
    train_filename = "train.tsv"
    valid_filename = "dev.tsv"
    test_filename = "test.tsv"
    unk_token = '<unk>'
    bos_token = '<s>'
    eos_token = '</s>'

    def __init__(self, segment="train", root=None):
        if segment not in ("train", "dev", "test"):
            raise ValueError("Only train|dev|test mode is supported.")
        data_path = CoupletDataset.get_data(root)
        self.vocab = CoupletDataset.build_vocab(data_path)
        filename_dict = {
            "train": self.train_filename,
            "dev": self.valid_filename,
            "test": self.test_filename
        }
        self.data = self.read_raw_data(
            os.path.join(data_path, filename_dict[segment]))
        self.transform()

    @classmethod
    def get_data(cls, root=None):
        if root is None:
            root = os.path.join(DATA_HOME, 'text_generation')
            data_dir = os.path.join(root, cls.dataset_dirname)
        if not os.path.exists(root):
            os.makedirs(root)
            print("IWSLT will be downloaded at ", root)
            get_path_from_url(cls.URL, root)
            print("Downloaded success......")
        else:
            filename_list = [
                cls.train_filename, cls.valid_filename, cls.test_filename,
                cls.vocab_filename
            ]
            for filename in filename_list:
                file_path = os.path.join(data_dir, filename)
                if not os.path.exists(file_path):
                    print(
                        "The dataset is incomplete and will be re-downloaded.")
                    get_path_from_url(cls.URL, root)
                    print("Downloaded success......")
                    break
        return data_dir

    @classmethod
    def build_vocab(cls, data_path=None, reverse=False):
        # Get vocab_func
        if data_path is None:
            data_path = os.path.join(
                os.path.join(DATA_HOME, 'text_generation'), cls.dataset_dirname)
        file_path = os.path.join(data_path, cls.vocab_filename)

        vocab = Vocab.load_vocabulary(file_path, cls.unk_token, cls.bos_token,
                                      cls.eos_token)
        return vocab._token_to_idx if not reverse else vocab._idx_to_token

    def read_raw_data(self, corpus_path):
        """Read raw files, return raw data"""
        data = []
        (f_mode, f_encoding, endl) = ("r", "utf-8", "\n")
        with io.open(corpus_path, f_mode, encoding=f_encoding) as f_corpus:
            for line in f_corpus.readlines():
                left = line.split("\t")[0]
                right = line.split("\t")[1]
                data.append((left, right))
        return data

    def transform(self):
        def vocab_func(vocab, unk_token):
            def func(tok_iter):
                return [
                    vocab[tok] if tok in vocab else vocab[unk_token]
                    for tok in tok_iter
                ]

            return func

        self.data = [([self.vocab[self.bos_token]] + vocab_func(
            self.vocab, self.unk_token)(data[0].split("\x02")) +
                      [self.vocab[self.eos_token]],
                      [self.vocab[self.bos_token]] + vocab_func(
                          self.vocab, self.unk_token)(data[1].split("\x02")) +
                      [self.vocab[self.eos_token]]) for data in self.data]

    def get_vocab(self):
        return self.vocab

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
