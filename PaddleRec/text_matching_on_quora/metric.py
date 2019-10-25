# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import numpy as np
"""
This Module defines evaluate metrics for classification tasks
"""


def accuracy(y_pred, label):
    """
    define correct: the top 1 class in y_pred is the same as y_true
    """
    y_pred = np.squeeze(y_pred)
    y_pred_idx = np.argmax(y_pred, axis=1)
    return 1.0 * np.sum(y_pred_idx == label) / label.shape[0]


def accuracy_with_threshold(y_pred, label, threshold=0.5):
    """
    define correct: the y_true class's prob in y_pred is bigger than threshold
    when threshold is 0.5, This fuction is equal to accuracy
    """
    y_pred = np.squeeze(y_pred)
    y_pred_idx = (y_pred[:, 1] > threshold).astype(int)
    return 1.0 * np.sum(y_pred_idx == label) / label.shape[0]
