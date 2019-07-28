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

from __future__ import division

import codecs

from mmpms.inputters.constant import UNK, BOS, EOS


class Vocabulary(object):

    unk_token = UNK
    bos_token = BOS
    eos_token = EOS

    def __init__(self, min_count=0, max_size=None, specials=[],
                 embed_file=None):
        self.min_count = min_count
        self.max_size = max_size
        self.embed_file = embed_file

        self.specials = [self.unk_token, self.bos_token, self.eos_token]
        for token in specials:
            if token not in self.specials:
                self.specials.append(token)

        self.itos = []
        self.stoi = {}
        self.embeddings = None

    @property
    def unk_id(self):
        return self.stoi.get(self.unk_token)

    @property
    def bos_id(self):
        return self.stoi.get(self.bos_token)

    @property
    def eos_id(self):
        return self.stoi.get(self.eos_token)

    def __len__(self):
        return len(self.itos)

    def size(self):
        return len(self.itos)

    def build(self, counter):
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]

        if len(counter) == 0:
            return

        self.itos = list(self.specials)

        if self.max_size is not None:
            self.max_size += len(self.itos)

        # sort by frequency, then alphabetically
        tokens_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        tokens_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        cover = 0
        for token, count in tokens_frequencies:
            if count < self.min_count or len(self.itos) == self.max_size:
                break
            self.itos.append(token)
            cover += count
        cover = cover / sum(count for _, count in tokens_frequencies)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

        print("Built vocabulary of size {} ".format(self.size()) +
              "(coverage: {:.3f})".format(cover))

        if self.embed_file is not None:
            self.embeddings = self.build_word_embeddings(self.embed_file)

    def build_word_embeddings(self, embed_file):
        cover = 0
        print("Building word embeddings from '{}' ...".format(embed_file))
        with codecs.open(embed_file, "r", encoding="utf-8") as f:
            num, dim = map(int, f.readline().strip().split())
            embeds = [[0] * dim] * len(self.stoi)
            for line in f:
                cols = line.rstrip().split()
                w, vs = cols[0], cols[1:]
                if w in self.stoi:
                    try:
                        vs = [float(x) for x in vs]
                    except Exception:
                        vs = []
                    if len(vs) == dim:
                        embeds[self.stoi[w]] = vs
                        cover += 1
        rate = cover / len(embeds)
        print("Built {} {}-D pretrained word embeddings ".format(cover, dim) +
              "(coverage: {:.3f})".format(rate))
        return embeds

    def dump(self):
        vocab_dict = {"itos": self.itos, "embeddings": self.embeddings}
        return vocab_dict

    def load(self, vocab_dict):
        self.itos = vocab_dict["itos"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.embeddings = vocab_dict["embeddings"]
