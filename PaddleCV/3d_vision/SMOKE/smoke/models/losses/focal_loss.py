# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn


class FocalLoss(nn.Layer):
    """Focal loss class
    """
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        """forward

        Args:
            prediction (paddle.Tensor): model prediction
            target (paddle.Tensor): ground truth

        Returns:
            paddle.Tensor: focal loss
        """
        positive_index = (target == 1).astype("float32")
        negative_index = (target < 1).astype("float32")

        negative_weights = paddle.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = paddle.log(prediction) \
                        * paddle.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = paddle.log(1 - prediction) \
                        * paddle.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive

        return loss
