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

import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['ProtoTypical', 'AMSoftmaxLoss', 'CMSoftmax']


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
                 margin: float = 0.3,
                 scale: float = 30.0):
        super(AMSoftmaxLoss, self).__init__()
        self.w = paddle.create_parameter((feature_dim, n_classes), 'float32')
        self.eps = eps
        self.scale = scale
        self.margin = margin
        self.nll_loss = nn.NLLLoss()
        self.n_classes = n_classes

    def forward(self, logits, label):
        logits = F.normalize(logits, p=2, axis=1, epsilon=self.eps)
        wn = F.normalize(self.w, p=2, axis=0, epsilon=self.eps)
        cosine = paddle.matmul(logits, wn)
        y = paddle.zeros((logits.shape[0], self.n_classes))
        for i in range(logits.shape[0]):
            y[i, label[i]] = self.margin
        pred = F.log_softmax((cosine - y) * self.scale, -1)
        return self.nll_loss(pred, label), pred


class ProtoTypical(nn.Layer):
    """Proto-typical loss as described in [1].

    Reference:
        [1] Chung, Joon Son, et al. “In Defence of Metric Learning for Speaker Recognition.”
         Interspeech 2020, 2020, pp. 2977–2981.

    """
    def __init__(self, s=20.0, eps=1e-8):
        super(ProtoTypical, self).__init__()
        self.nll_loss = nn.NLLLoss()
        self.eps = eps
        self.s = s

    def forward(self, logits):
        assert logits.ndim == 3, (
            f'the input logits must be a ' +
            f'3d tensor of shape [n_spk,n_uttns,emb_dim],' +
            f'but received logits.ndim = {logits.ndim}')
        import pdb
        pdb.set_trace()

        logits = F.normalize(logits, p=2, axis=-1, epsilon=self.eps)
        proto = paddle.mean(logits[:, 1:, :], axis=1, keepdim=False).transpose(
            (1, 0))  # [emb_dim, n_spk]
        query = logits[:, 0, :]  # [n_spk, emb_dim]
        similarity = paddle.matmul(query, proto) * self.s  #[n_spk,n_spk]
        label = paddle.arange(0, similarity.shape[0])
        log_sim = F.log_softmax(similarity, -1)
        return self.nll_loss(log_sim, label), log_sim


class AngularMargin(nn.Layer):
    def __init__(self, margin=0.0, scale=1.0):
        super(AngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        outputs = outputs - self.margin * targets
        return self.scale * outputs


class LogSoftmaxWrapper(nn.Layer):
    def __init__(self, loss_fn):
        super(LogSoftmaxWrapper, self).__init__()
        self.loss_fn = loss_fn
        self.criterion = paddle.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        targets = F.one_hot(targets, outputs.shape[1])
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, axis=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


class AdditiveAngularMargin(AngularMargin):
    def __init__(self,
                 margin=0.0,
                 scale=1.0,
                 feature_dim=256,
                 n_classes=1000,
                 easy_margin=False):
        super(AdditiveAngularMargin, self).__init__(margin, scale)
        self.easy_margin = easy_margin
        self.w = paddle.create_parameter((feature_dim, n_classes), 'float32')
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.nll_loss = nn.NLLLoss()
        self.n_classes = n_classes

    def forward(self, logits, targets):
        # logits = self.drop(logits)
        logits = F.normalize(logits, p=2, axis=1, epsilon=1e-8)
        wn = F.normalize(self.w, p=2, axis=0, epsilon=1e-8)
        cosine = logits @ wn

        #cosine = outputs.astype('float32')
        sine = paddle.sqrt(1.0 - paddle.square(cosine))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)
        target_one_hot = F.one_hot(targets, self.n_classes)
        outputs = (target_one_hot * phi) + ((1.0 - target_one_hot) * cosine)
        outputs = self.scale * outputs
        pred = F.log_softmax(outputs, axis=-1)

        return self.nll_loss(pred, targets), pred


class CMSoftmax(AngularMargin):
    def __init__(self,
                 margin=0.0,
                 margin2=0.0,
                 scale=1.0,
                 feature_dim=256,
                 n_classes=1000,
                 easy_margin=False):
        super(CMSoftmax, self).__init__(margin, scale)
        self.easy_margin = easy_margin
        self.w = paddle.create_parameter((feature_dim, n_classes), 'float32')
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.nll_loss = nn.NLLLoss()
        self.n_classes = n_classes
        self.margin2 = margin2

    def forward(self, logits, targets):
        logits = F.normalize(logits, p=2, axis=1, epsilon=1e-8)
        wn = F.normalize(self.w, p=2, axis=0, epsilon=1e-8)
        cosine = logits @ wn

        sine = paddle.sqrt(1.0 - paddle.square(cosine))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)
        target_one_hot = F.one_hot(targets, self.n_classes)
        outputs = (target_one_hot * phi) + (
            (1.0 - target_one_hot) * cosine) - target_one_hot * self.margin2
        outputs = self.scale * outputs
        pred = F.log_softmax(outputs, axis=-1)

        return self.nll_loss(pred, targets), pred
