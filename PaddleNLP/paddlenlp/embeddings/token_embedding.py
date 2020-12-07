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

from enum import Enum
import os
import os.path as osp
import numpy as np
import jieba

import paddle
import paddle.nn as nn
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME, _get_sub_home
from .constant import *

EMBEDDING_HOME = _get_sub_home('embedding', parent_home=DATA_HOME)


def get_corpus_path(corpus_name):
    return CORPUS_NAME_MAP[corpus_name]


def _get_idx_from_word(word, word_to_idx):
    if word in word_to_idx:
        return word_to_idx[word]
    return word_to_idx[UNK_WORD]


class BaseEmbeddingTokenizer(object):
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self._word_to_idx = {PAD_WORD: PAD_IDX, UNK_WORD: UNK_IDX}
        with open(vocab_path) as f:
            idx = 2
            for line in f.readlines():
                if line.strip() == "":
                    break
                word, freq = line.split()
                self._word_to_idx[word] = idx
                idx += 1

    def get_tokenizer(self):
        return self.tokenizer

    def cut(self, sentence):
        pass

    def encode(self, sentence):
        pass


class JiebaEmbeddingTokenizer(BaseEmbeddingTokenizer):
    def __init__(self, vocab_path):
        super(JiebaEmbeddingTokenizer, self).__init__(vocab_path)
        self.tokenizer = jieba.Tokenizer(vocab_path)

    def cut(self, sentence, cut_all=False, HMM=True):
        return [word for word in self.tokenizer.cut(sentence, cut_all, HMM)]

    def encode(self, sentence, cut_all=False, HMM=True):
        words = self.cut(sentence, cut_all, HMM)
        return [_get_idx_from_word(word, self._word_to_idx) for word in words]


class TokenEmbedding(nn.Embedding):
    def __init__(self, corpus_name=CorpusName.SOGOU_NEWS, trainable=True):
        if isinstance(corpus_name, str):
            corpus_name = CorpusName[corpus_name]
        else:
            corpus_name = CorpusName(corpus_name)

        corpus_path = get_corpus_path(corpus_name)
        vector_path = osp.join(EMBEDDING_HOME, corpus_path + ".tar",
                               corpus_path + ".npz")
        self.vocab_path = osp.join(EMBEDDING_HOME, corpus_path + ".tar",
                                   corpus_path + ".txt")
        if not osp.exists(vector_path):
            # download
            url = URL_ROOT + "/" + corpus_path + ".tar.gz"
            get_path_from_url(url, EMBEDDING_HOME)

        vector_np = np.load(vector_path)
        self._idx_to_word = vector_np['vocab']
        self._word_to_idx = self._construct_word_to_idx(self._idx_to_word)

        # import embedding
        self.num_embeddings, self.embedding_dim = vector_np['embedding'].shape
        super(TokenEmbedding, self).__init__(
            self.num_embeddings, self.embedding_dim, padding_idx=0)
        self.weight.set_value(vector_np['embedding'])
        self.set_trainable(trainable)

    def set_trainable(self, trainable):
        self.weight.stop_gradient = not trainable

    def search(self, words):
        idx_list = self.get_idx_list_from_words(words)
        idx_tensor = paddle.to_tensor(idx_list)
        return self(idx_tensor).numpy()

    def get_idx_from_word(self, word):
        return _get_idx_from_word(word, self._word_to_idx)

    def get_idx_list_from_words(self, words):
        if isinstance(words, str):
            idx_list = [self.get_idx_from_word(words)]
        elif isinstance(words, int):
            idx_list = [words]
        elif isinstance(words, list) or isinstance(words, tuple):
            idx_list = [
                self.get_idx_from_word(word) if isinstance(word, str) else word
                for word in words
            ]
        else:
            raise TypeError
        return idx_list

    def cosine_sim(self, word_a, word_b):
        embeddings = self.search([word_a, word_b])
        embedding_a = embeddings[0]
        embedding_b = embeddings[1]

        def dot(a, b):
            return np.sum(a * b)
        return dot(embedding_a, embedding_b) /            \
            (np.sqrt(dot(embedding_a, embedding_a)) *     \
             np.sqrt(dot(embedding_b, embedding_b)))

    def _construct_word_to_idx(self, idx_to_word):
        word_to_idx = {}
        for i, word in enumerate(idx_to_word):
            word_to_idx[word] = i
        return word_to_idx
