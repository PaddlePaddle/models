#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import math

import numpy as np
import paddle
import paddle.nn as nn

from paddlenlp.layers.crf import LinearChainCrf, ViterbiDecoder


class BiGruCrf(nn.Layer):
    """The network for lexical analysis, based on two layers of BiGRU and one layer of CRF. More details see https://arxiv.org/abs/1807.01882

    Args:
        word_emb_dim (int): The dimension in which a word is embedded.
        hidden_size (int): The number of hidden nodes in the GRU layer.
        vocab_size (int): the word vocab size.
        num_labels (int): the labels amount.
        emb_lr (float, optional): The scaling of the learning rate of the embedding layer. Defaults to 2.0.
        crf_lr (float, optional): The scaling of the learning rate of the crf layer. Defaults to 0.2.
    """

    def __init__(self,
                 word_emb_dim,
                 hidden_size,
                 vocab_size,
                 num_labels,
                 emb_lr=2.0,
                 crf_lr=0.2):
        super(BiGruCrf, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.emb_lr = emb_lr
        self.crf_lr = crf_lr
        self.init_bound = 0.1

        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_emb_dim,
            weight_attr=paddle.ParamAttr(
                learning_rate=self.emb_lr,
                initializer=nn.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound)))

        self.gru = nn.GRU(
            input_size=self.word_emb_dim,
            hidden_size=self.hidden_size,
            num_layers=2,
            direction='bidirectional',
            weight_ih_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)),
            weight_hh_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)))

        self.fc = nn.Linear(
            in_features=self.hidden_size * 2,
            out_features=self.num_labels + 2,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(
                    low=-self.init_bound, high=self.init_bound),
                regularizer=paddle.regularizer.L2Decay(coeff=1e-4)))

        self.crf = LinearChainCrf(self.num_labels, self.crf_lr)
        self.viterbi_decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, inputs, lengths):
        word_embed = self.word_embedding(inputs)
        bigru_output, _ = self.gru(word_embed)
        emission = self.fc(bigru_output)
        _, prediction = self.viterbi_decoder(emission, lengths)
        return emission, lengths, prediction


class ChunkEvaluator(paddle.metric.Metric):
    """ChunkEvaluator computes the precision, recall and F1-score for chunk detection.
    It is often used in sequence tagging tasks, such as Named Entity Recognition(NER).

    Args:
        num_chunk_types (int): The number of chunk types.
        chunk_scheme (str): Indicate the tagging schemes used here. The value must
            be IOB, IOE, IOBES or plain.
        excluded_chunk_types (list, optional): Indicate the chunk types shouldn't
            be taken into account. It should be a list of chunk type ids(integer).
            Default None.
    """

    def __init__(self, num_chunk_types, chunk_scheme,
                 excluded_chunk_types=None):
        super(ChunkEvaluator, self).__init__()
        self.num_chunk_types = num_chunk_types
        self.chunk_scheme = chunk_scheme
        self.excluded_chunk_types = excluded_chunk_types
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def compute(self, inputs, lengths, predictions, labels):
        precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks = paddle.metric.chunk_eval(
            predictions,
            labels,
            chunk_scheme=self.chunk_scheme,
            num_chunk_types=self.num_chunk_types,
            excluded_chunk_types=self.excluded_chunk_types,
            seq_length=lengths)

        return num_infer_chunks, num_label_chunks, num_correct_chunks

    def _is_number_or_matrix(self, var):
        def _is_number_(var):
            return isinstance(
                var, int) or isinstance(var, np.int64) or isinstance(
                    var, float) or (isinstance(var, np.ndarray) and
                                    var.shape == (1, ))

        return _is_number_(var) or isinstance(var, np.ndarray)

    def update(self, num_infer_chunks, num_label_chunks, num_correct_chunks):
        """
        This function takes (num_infer_chunks, num_label_chunks, num_correct_chunks) as input,
        to accumulate and update the corresponding status of the ChunkEvaluator object. The update method is as follows:

        .. math::
                   \\\\ \\begin{array}{l}{\\text { self. num_infer_chunks }+=\\text { num_infer_chunks }} \\\\ {\\text { self. num_Label_chunks }+=\\text { num_label_chunks }} \\\\ {\\text { self. num_correct_chunks }+=\\text { num_correct_chunks }}\\end{array} \\\\

        Args:
            num_infer_chunks(int|numpy.array): The number of chunks in Inference on the given minibatch.
            num_label_chunks(int|numpy.array): The number of chunks in Label on the given mini-batch.
            num_correct_chunks(int|float|numpy.array): The number of chunks both in Inference and Label on the
                                                  given mini-batch.
        """
        if not self._is_number_or_matrix(num_infer_chunks):
            raise ValueError(
                "The 'num_infer_chunks' must be a number(int) or a numpy ndarray."
            )
        if not self._is_number_or_matrix(num_label_chunks):
            raise ValueError(
                "The 'num_label_chunks' must be a number(int, float) or a numpy ndarray."
            )
        if not self._is_number_or_matrix(num_correct_chunks):
            raise ValueError(
                "The 'num_correct_chunks' must be a number(int, float) or a numpy ndarray."
            )
        self.num_infer_chunks += num_infer_chunks
        self.num_label_chunks += num_label_chunks
        self.num_correct_chunks += num_correct_chunks

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            float: mean precision, recall and f1 score.
        """
        precision = float(
            self.num_correct_chunks
        ) / self.num_infer_chunks if self.num_infer_chunks else 0
        recall = float(self.num_correct_chunks
                       ) / self.num_label_chunks if self.num_label_chunks else 0
        f1_score = float(2 * precision * recall) / (
            precision + recall) if self.num_correct_chunks else 0
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_chunks = 0
        self.num_label_chunks = 0
        self.num_correct_chunks = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"
