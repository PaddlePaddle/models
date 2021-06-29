# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F


class AMSoftmaxLoss(nn.Layer):
    """Additive margin softmax loss.
    Additive margin softmax loss is usefully for training neural networks for speaker recognition/verification.

    Notes:
        The loss itself contains parameters that need to pass to optimizer for gradient descends.

    References:
        Wang, Feng, et al. “Additive Margin Softmax for Face Verification.”
        IEEE Signal Processing Letters, vol. 25, no. 7, 2018, pp. 926–930.

    """
    def __init__(self,
                 feature_dim: int,
                 n_classes: int,
                 eps: float = 1e-5,
                 m: float = 0.3,
                 s: float = 30.0):
        super(AMSoftmaxLoss, self).__init__()
        self.w = paddle.create_parameter((feature_dim, n_classes), 'float32')
        self.eps = eps
        self.s = s
        self.m = m
        self.nll_loss = nn.NLLLoss()
        self.n_classes = n_classes

    def forward(self, logit, label):
        logit = F.normalize(logit, p=2, axis=1, epsilon=self.eps)
        wn = F.normalize(self.w, p=2, axis=0, epsilon=self.eps)
        cosine = paddle.matmul(logit, wn)
        y = paddle.zeros((logit.shape[0], self.n_classes))
        for i in range(logit.shape[0]):
            y[i, label[i]] = self.m
        pred = F.log_softmax((cosine - y) * self.s, -1)
        return self.nll_loss(pred, label), pred


if __name__ == '__main__':
    loss = AMSoftmaxLoss(512, 10)
    print(loss.parameters())
