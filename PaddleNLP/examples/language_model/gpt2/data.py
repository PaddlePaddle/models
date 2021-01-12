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
import json
import random
from bisect import bisect_right
from itertools import accumulate

import nltk
import numpy as np
import pandas as pd
import paddle


class GPT2Dataset(paddle.io.Dataset):
    def __init__(self,
                 file_path,
                 tokenizer,
                 num_samples=None,
                 max_seq_len=1024,
                 weighted=True,
                 sample_across_doc=True,
                 random_across_doc_sampling=True,
                 reset_attenion_mask=False,
                 reset_position_id=False,
                 sentence_start=False,
                 mode="train"):
        self.file_path = file_path
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.weighted = weighted
        self.sample_across_doc = sample_across_doc
        self.random_across_doc_sampling = random_across_doc_sampling
        self.reset_attenion_mask = reset_attenion_mask
        self.reset_position_id = reset_position_id
        self.sentence_start = sentence_start

        self.example_texts = []
        self._read_json()
        self.num_example_texts = len(self.example_texts)
        if num_samples is None:
            self.num_samples = 5 * self.num_example_texts
        self.eos_id = tokenizer.get_command("eos").Id
        print("the eos is:{}".format(self.eos_id))

        self._init_weighting()

    def _read_json(self):
        nltk.download("punkt")
        with open(self.file_path, "r") as input_file:
            for line in input_file.readlines():
                self.example_texts.append(json.loads(line)['text'])

    def _init_weighting(self):
        if self.weighted:
            lens = np.array([len(d) for d in self.example_texts])
            self.total_len = np.sum(lens)
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None

    def _get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(0, self.num_example_texts)

    def _pad_seq(self, seq):
        total_tokens = self.max_seq_len + 1
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.get_command('pad').Id] * (num_pad_tokens)
        return seq

    def _getidx(self, data_idx):
        data = self.example_texts[data_idx]
        # tokenize
        tokenization = self.tokenizer.encode(data)
        tokenization.append(self.tokenizer.get_command('eos'))
        tokens = tokenization.tokenization
        return tokens

    def _contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        return False

    def _construct_sample(self, tokens):
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

        # the pad and eod tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[np.where(np.array(tokens) == self.eos_id)] = 0.0
        position_ids = np.arange(0, seq_length, dtype="int64")

        if self.reset_attenion_mask or self.reset_position_id:
            eos_indices = position_ids[np.where(tokens == self.eos_id)]
            prev_index = 0
            for i in range(eos_indices.size()[0]):
                pos_id = eos_indices[i]
                if self.reset_attention_mask:
                    attention_mask[0, (pos_id + 1):, :(pos_id + 1)] = 0
                if self.reset_position_ids:
                    position_ids[(pos_id + 1):] -= (pos_id + 1 - prev_index)
                    prev_index = i + 1
        attention_mask = (attention_mask - 1.0) * 10000.0
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, index):
        # init rng
        rng = random.Random(index)
        rng = np.random.RandomState(
            seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        data_idx = self._get_weighted_samples(rng)
        tokens = self._getidx(data_idx)

        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len - 1
        if tokens_to_strip > 0:
            strip_left_tokens = rng.randint(tokens_to_strip + 1)
            tokens = tokens[strip_left_tokens:]
            if self.sentence_start:
                token_copy = list(tokens)
                not_done = True
                while (len(token_copy) > 0) and not_done:
                    tok = token_copy.pop(0)
                    if self._contains_sentence_end(tok):
                        tokens = token_copy
                        not_done = False
            strip_right_rokens = len(tokens) - self.max_seq_len - 1
            if strip_right_rokens > 0:
                tokens = tokens[:-strip_right_rokens]

        if self.sample_across_doc:
            while (len(tokens) < (self.max_seq_len + 1)):
                if self.random_across_doc_sampling:
                    data_idx = self._get_weighted_samples(rng)
                else:
                    data_idx = (data_idx + 1) % self.num_example_texts
                tokens += self._getidx(data_idx)
            tokens = tokens[:(self.max_seq_len + 1)]

        tokens = self._pad_seq(tokens)
        return self._construct_sample(tokens)

    def __len__(self):
        return self.num_samples
