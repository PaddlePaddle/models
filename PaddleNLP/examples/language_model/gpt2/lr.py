# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import math
import numpy
import warnings
from paddle import Tensor
from paddle.optimizer.lr import LRScheduler


class CosineAnnealingWithWarmupDecay(LRScheduler):
    def __init__(self,
                 learning_rate,
                 warmup_iter,
                 num_iters,
                 last_epoch=-1,
                 verbose=False):

        self.end_iter = num_iters
        self.warmup_iter = warmup_iter
        super(CosineAnnealingWithWarmupDecay, self).__init__(
            learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.warmup_iter > 0 and self.last_epoch <= self.warmup_iter:
            return float(self.base_lr) * (self.last_epoch) / self.warmup_iter
        return self.base_lr / 2.0 * (
            math.cos(math.pi *
                     (self.last_epoch - self.warmup_iter) / self.end_iter) + 1)
