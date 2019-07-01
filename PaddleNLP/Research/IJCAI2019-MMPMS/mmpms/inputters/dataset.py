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

import codecs
import re
import time
import json
import pickle
from collections import Counter

from mmpms.inputters.vocabulary import Vocabulary
from mmpms.inputters.constant import UNK, BOS, EOS, NUM


def tokenize(s):
    s = re.sub('\d+', NUM, s).lower()
    tokens = s.split(' ')
    return tokens


class PostResponseDataset(object):
    def __init__(self,
                 tokenize_fn=tokenize,
                 min_count=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None):
        self.tokenize_fn = tokenize_fn
        self.vocab = Vocabulary(
            min_count=min_count, max_size=max_vocab_size, embed_file=embed_file)
        self.min_len = min_len
        self.max_len = max_len

    def build_vocab(self, data_file):
        examples = self.read(data_file)
        counter = Counter()
        print("Building vocabulary ...")
        for example in examples:
            counter.update(example["post"])
            counter.update(example["response"])
        self.vocab.build(counter)

    def save_vocab(self, vocab_file):
        vocab_dict = self.vocab.dump()
        start = time.time()
        with codecs.open(vocab_file, "w", encoding="utf-8") as fp:
            json.dump(vocab_dict, fp, ensure_ascii=False)
        elapsed = time.time() - start
        print("Saved vocabulary to '{}' (elapsed {:.2f}s)".format(vocab_file,
                                                                  elapsed))

    def load_vocab(self, vocab_file):
        print("Loading vocabulary from '{}' ...".format(vocab_file))
        start = time.time()
        with codecs.open(vocab_file, "r", encoding="utf-8") as fp:
            vocab_dict = json.load(fp)
        elapsed = time.time() - start
        self.vocab.load(vocab_dict)
        vocab_size = self.vocab.size()
        print("Loaded vocabulary of size {} (elapsed {}s)".format(vocab_size,
                                                                  elapsed))

    def indices2string(self, indices):
        tokens = [self.vocab.itos[idx] for idx in indices]

        bos_token = self.vocab.bos_token
        if bos_token and tokens[0] == bos_token:
            tokens = tokens[1:]

        eos_token = self.vocab.eos_token
        string = []
        for tok in tokens:
            if tok != eos_token:
                string.append(tok)
            else:
                break
        string = " ".join(string)
        return string

    def tokens2indices(self, tokens):
        indices = [
            self.vocab.stoi.get(tok, self.vocab.unk_id) for tok in tokens
        ]
        return indices

    def numericalize(self, tokens):
        element = tokens[0]
        if isinstance(element, list):
            return [self.numericalize(s) for s in tokens]
        else:
            return self.tokens2indices(tokens)

    def denumericalize(self, indices):
        element = indices[0]
        if isinstance(element, list):
            return [self.denumericalize(x) for x in indices]
        else:
            return self.indices2string(indices)

    def build_examples(self, data_file):
        print("Building examples from '{}' ...".format(data_file))
        data = self.read(data_file)
        examples = []
        print("Numericalizing examples ...")
        for ex in data:
            example = {}
            post, response = ex["post"], ex["response"]
            post = self.numericalize(post)
            response = self.numericalize(response)
            example["post"] = post
            example["response"] = [self.vocab.bos_id] + response
            example["label"] = response + [self.vocab.eos_id]
            examples.append(example)
        return examples

    def save_examples(self, examples, filename):
        start = time.time()
        with open(filename, "wb") as fp:
            pickle.dump(examples, fp)
        elapsed = time.time() - start
        print("Saved examples to '{}' (elapsed {:.2f}s)".format(filename,
                                                                elapsed))

    def load_examples(self, filename):
        print("Loading examples from '{}' ...".format(filename))
        start = time.time()
        with open(filename, "rb") as fp:
            examples = pickle.load(fp)
        elapsed = time.time() - start
        print("Loaded {} examples (elapsed {:.2f}s)".format(
            len(examples), elapsed))
        return examples

    def read(self, data_file):
        examples = []
        ignored = 0

        def filter_pred(utt):
            """ Filter utterance. """
            return self.min_len <= len(utt) <= self.max_len

        print("Reading examples from '{}' ...".format(data_file))
        with codecs.open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                post, response = line.strip().split("\t")
                post = self.tokenize_fn(post)
                response = self.tokenize_fn(response)
                if filter_pred(post) and filter_pred(response):
                    examples.append({"post": post, "response": response})
                else:
                    ignored += 1
        print("Read {} examples ({} filtered)".format(len(examples), ignored))
        return examples
