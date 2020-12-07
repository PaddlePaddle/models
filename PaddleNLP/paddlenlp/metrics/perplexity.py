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


class Perplexity(paddle.metric.Metric):
    def __init__(self, name='Perplexity', *args, **kwargs):
        super(Perplexity, self).__init__(*args, **kwargs)
        self._name = name
        self.total_ce = 0
        self.num_batch = 0

    def update(self, y, label, *args):
        # Perplexity is calculated using cross entropy
        label = paddle.to_tensor(label)
        y = paddle.to_tensor(y)
        label = paddle.unsqueeze(label, axis=2)

        ce = paddle.nn.functional.softmax_with_cross_entropy(
            logits=y, label=label, soft_label=False)
        ce = paddle.mean(ce)

        self.total_ce += ce.numpy()[0]
        self.num_batch += 1

    def reset(self):
        self.total_ce = 0
        self.num_batch = 0

    def accumulate(self):
        return np.exp(self.total_ce / self.num_batch)

    def name(self):
        return self._name
