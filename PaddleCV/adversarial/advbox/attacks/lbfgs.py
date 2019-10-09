#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
This module provide the attack method of "LBFGS".
"""
from __future__ import division

import logging

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .base import Attack

__all__ = ['LBFGSAttack', 'LBFGS']


class LBFGSAttack(Attack):
    """
    Uses L-BFGS-B to minimize the cross-entropy and the distance between the
    original and the adversary.

    Paper link: https://arxiv.org/abs/1510.05328
    """

    def __init__(self, model):
        super(LBFGSAttack, self).__init__(model)
        self._predicts_normalized = None
        self._adversary = None  # type: Adversary

    def _apply(self, adversary, epsilon=0.001, steps=10):
        self._adversary = adversary

        if not adversary.is_targeted_attack:
            raise ValueError("This attack method only support targeted attack!")

        # finding initial c
        logging.info('finding initial c...')
        c = epsilon
        x0 = adversary.original.flatten()
        for i in range(30):
            c = 2 * c
            logging.info('c={}'.format(c))
            is_adversary = self._lbfgsb(x0, c, steps)
            if is_adversary:
                break
        if not is_adversary:
            logging.info('Failed!')
            return adversary

        # binary search c
        logging.info('binary search c...')
        c_low = 0
        c_high = c
        while c_high - c_low >= epsilon:
            logging.info('c_high={}, c_low={}, diff={}, epsilon={}'
                         .format(c_high, c_low, c_high - c_low, epsilon))
            c_half = (c_low + c_high) / 2
            is_adversary = self._lbfgsb(x0, c_half, steps)
            if is_adversary:
                c_high = c_half
            else:
                c_low = c_half

        return adversary

    def _is_predicts_normalized(self, predicts):
        """
        To determine the predicts is normalized.
        :param predicts(np.array): the output of the model.
        :return: bool
        """
        if self._predicts_normalized is None:
            if self.model.predict_name().lower() in [
                    'softmax', 'probabilities', 'probs'
            ]:
                self._predicts_normalized = True
            else:
                if np.any(predicts < 0.0):
                    self._predicts_normalized = False
                else:
                    s = np.sum(predicts.flatten())
                    if 0.999 <= s <= 1.001:
                        self._predicts_normalized = True
                    else:
                        self._predicts_normalized = False
        assert self._predicts_normalized is not None
        return self._predicts_normalized

    def _loss(self, adv_x, c):
        """
        To get the loss and gradient.
        :param adv_x: the candidate adversarial example
        :param c: parameter 'C' in the paper
        :return: (loss, gradient)
        """
        x = adv_x.reshape(self._adversary.original.shape)

        # cross_entropy
        logits = self.model.predict(x)
        if not self._is_predicts_normalized(logits):  # to softmax
            e = np.exp(logits)
            logits = e / np.sum(e)
        e = np.exp(logits)
        s = np.sum(e)
        ce = np.log(s) - logits[self._adversary.target_label]

        # L2 distance
        min_, max_ = self.model.bounds()
        d = np.sum((x - self._adversary.original).flatten() ** 2) \
            / ((max_ - min_) ** 2) / len(adv_x)

        # gradient
        gradient = self.model.gradient(x, self._adversary.target_label)

        result = (c * ce + d).astype(float), gradient.flatten().astype(float)
        return result

    def _lbfgsb(self, x0, c, maxiter):
        min_, max_ = self.model.bounds()
        bounds = [(min_, max_)] * len(x0)
        approx_grad_eps = (max_ - min_) / 100.0
        x, f, d = fmin_l_bfgs_b(
            self._loss,
            x0,
            args=(c, ),
            bounds=bounds,
            maxiter=maxiter,
            epsilon=approx_grad_eps)
        if np.amax(x) > max_ or np.amin(x) < min_:
            x = np.clip(x, min_, max_)
        shape = self._adversary.original.shape
        adv_label = np.argmax(self.model.predict(x.reshape(shape)))
        logging.info('pre_label = {}, adv_label={}'.format(
            self._adversary.target_label, adv_label))
        return self._adversary.try_accept_the_example(
            x.reshape(shape), adv_label)


LBFGS = LBFGSAttack
