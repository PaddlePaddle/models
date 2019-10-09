#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

def accuracy(targets, preds):
    """Get the class-level top1 and top5 of model.

    Usage:

    .. code-blcok::python

        top1, top5 = accuracy(targets, preds)

    :params args: evaluate the prediction of model.
    :type args: numpy.array

    """
    top1 = np.zeros((5000,), dtype=np.float32)
    top5 = np.zeros((5000,), dtype=np.float32)
    count = np.zeros((5000,), dtype=np.float32)

    for index in range(targets.shape[0]):
        target = targets[index]
        if target == preds[index,0]:
            top1[target] += 1
            top5[target] += 1
        elif np.sum(target == preds[index,:5]):
            top5[target] += 1

        count[target] += 1
    return (top1/(count+1e-12)).mean(), (top5/(count+1e-12)).mean()
