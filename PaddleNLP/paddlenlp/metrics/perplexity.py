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

import paddle
import numpy as np

import paddle.nn.functional as F


class Perplexity(paddle.metric.Metric):
    """
    Perplexity is calculated using cross entropy. It supports both padding data
    and no padding data.

    If data is not padded, users should provide `seq_len` for `Metric`
    initialization. If data is padded, your label should contain `seq_mask`,
    which indicates the actual length of samples.

    This Perplexity requires that the output of your network is prediction,
    label and sequence length (opitonal). If the Perplexity here doesn't meet
    your needs, you could override the `compute` or `update` method for
    caculating Perplexity.

    Args:
        seq_len(int): Sequence length of each sample, it must be provided while
            data is not padded. Default: 20.
        name(str): Name of `Metric` instance. Default: 'Perplexity'.

    """

    def __init__(self, seq_len=20, name='Perplexity', *args, **kwargs):
        super(Perplexity, self).__init__(*args, **kwargs)
        self._name = name
        self.total_cost = 0
        self.word_count = 0
        self.word_num = seq_len

    def compute(self, pred, label, seq_mask=None):
        cost = F.softmax_with_cross_entropy(
            logits=pred, label=label, soft_label=False)
        cost = paddle.squeeze(cost, axis=[2])
        word_num = self.word_num

        if seq_mask is not None:
            cost = cost * seq_mask
            word_num = paddle.sum(seq_mask)
        batch_cost = paddle.sum(cost)
        return batch_cost, word_num

    def update(self, batch_cost, word_num):
        self.total_cost += batch_cost
        self.word_count += word_num

    def reset(self):
        self.total_cost = 0
        self.word_count = 0

    def accumulate(self):
        return np.exp(self.total_cost / self.word_count)

    def name(self):
        return self._name
