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

import paddle
import paddle.nn as nn
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import _get_sub_home, MODEL_HOME
from paddlenlp.data import Vocab, get_idx_from_word
from .constant import *

EMBEDDING_HOME = _get_sub_home('embeddings', parent_home=MODEL_HOME)


def list_embedding_name():
    return list(EMBEDDING_NAME_LIST)


class TokenEmbedding(nn.Embedding):
    def __init__(self,
                 embedding_name=EMBEDDING_NAME_LIST[0],
                 unknown_word=UNK_WORD,
                 unknown_word_vector=None,
                 extended_vocab_path=None,
                 trainable=True):

        embedding_path = embedding_name.lower()
        vector_path = osp.join(EMBEDDING_HOME, embedding_path + ".npz")
        if not osp.exists(vector_path):
            # download
            url = osp.join(EMBEDDING_URL_ROOT, embedding_path + ".tar.gz")
            get_path_from_url(url, EMBEDDING_HOME)

        self.unknown_word = unknown_word

        vector_np = np.load(vector_path)
        self._idx_to_word = list(vector_np['vocab'])
        self.embedding_dim = vector_np['embedding'].shape[1]
        if unknown_word_vector is not None:
            unk_vector = np.array(unknown_word_vector).astype(
                paddle.get_default_dtype())
        else:
            unk_vector = np.random.normal(
                scale=0.02,
                size=self.embedding_dim).astype(paddle.get_default_dtype())
        pad_vector = np.array(
            [0] * self.embedding_dim).astype(paddle.get_default_dtype())

        # insert unk, pad embedding
        embedding_table = np.insert(
            vector_np['embedding'], [0], [pad_vector, unk_vector],
            axis=0).astype(paddle.get_default_dtype())
        self._idx_to_word.insert(PAD_IDX, PAD_WORD)
        self._idx_to_word.insert(UNK_IDX, self.unknown_word)

        self._word_to_idx = self._construct_word_to_idx(self._idx_to_word)
        if extended_vocab_path is not None:
            new_words_embedding = self._extend_vocab(extended_vocab_path,
                                                     embedding_table)
            embedding_table = np.append(
                embedding_table, new_words_embedding, axis=0)
            trainable = True

        self.vocab = Vocab.from_dict(
            self._word_to_idx, unk_token=unknown_word, pad_token=PAD_WORD)
        self.num_embeddings = embedding_table.shape[0]
        # import embedding
        super(TokenEmbedding, self).__init__(
            self.num_embeddings, self.embedding_dim, padding_idx=PAD_IDX)
        self.weight.set_value(embedding_table)
        self.set_trainable(trainable)

    def _read_vocab_list_from_file(self, extended_vocab_path):
        # load new vocab table from file
        vocab_list = []
        with open(extended_vocab_path) as f:
            for line in f.readlines():
                if line.strip() == "":
                    break
                vocab_list.append(line)
        return vocab_list

    def _extend_vocab(self, extended_vocab_path, embedding_table):
        extend_vocab_list = self._read_vocab_list_from_file(extended_vocab_path)
        curr_idx = len(self._idx_to_word)
        new_words = list(
            set(word for word in extend_vocab_list
                if word not in self._word_to_idx))
        # update idx_to_word
        self._idx_to_word.extend(new_words)
        # update word_to_idx
        for i, word in enumerate(new_words):
            self._word_to_idx[word] = i + curr_idx
        # update embedding_table
        new_words_embedding = np.random.normal(
            scale=0.02,
            size=(len(new_words),
                  self.embedding_dim)).astype(paddle.get_default_dtype())
        return new_words_embedding

    def set_trainable(self, trainable):
        self.weight.stop_gradient = not trainable

    def search(self, words):
        idx_list = self.get_idx_list_from_words(words)
        idx_tensor = paddle.to_tensor(idx_list)
        return self(idx_tensor).numpy()

    def get_idx_from_word(self, word):
        return get_idx_from_word(word, self.vocab.token_to_idx,
                                 self.unknown_word)

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

    def get_unk_idx_word(self):
        return "Unknown index = {}, word = {}".format(UNK_IDX,
                                                      self.unknown_word)

    def get_pad_idx_word(self):
        return "Padding index = {}, word = {}".format(PAD_IDX, PAD_WORD)
