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
This module provide the attack method for deepfool. Deepfool is a simple and
accurate adversarial attack.
"""
from __future__ import division

import logging

import numpy as np

from .base import Attack

__all__ = ['DeepFoolAttack']


class DeepFoolAttack(Attack):
    """
    DeepFool: a simple and accurate method to fool deep neural networks",
    Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
    https://arxiv.org/abs/1511.04599
    """

    def _apply(self, adversary, iterations=100, overshoot=0.02):
        """
          Apply the deep fool attack.

          Args:
              adversary(Adversary): The Adversary object.
              iterations(int): The iterations.
              overshoot(float): We add (1+overshoot)*pert every iteration.
          Return:
              adversary: The Adversary object.
          """
        assert adversary is not None

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()
        f = self.model.predict(adversary.original)
        if adversary.is_targeted_attack:
            labels = [adversary.target_label]
        else:
            max_class_count = 10
            class_count = self.model.num_classes()
            if class_count > max_class_count:
                labels = np.argsort(f)[-(max_class_count + 1):-1]
            else:
                labels = np.arange(class_count)

        gradient = self.model.gradient(adversary.original, pre_label)
        x = adversary.original
        for iteration in xrange(iterations):
            w = np.inf
            w_norm = np.inf
            pert = np.inf
            for k in labels:
                if k == pre_label:
                    continue
                gradient_k = self.model.gradient(x, k)
                w_k = gradient_k - gradient
                f_k = f[k] - f[pre_label]
                w_k_norm = np.linalg.norm(w_k.flatten()) + 1e-8
                pert_k = (np.abs(f_k) + 1e-8) / w_k_norm
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                    w_norm = w_k_norm

            r_i = -w * pert / w_norm  # The gradient is -gradient in the paper.
            x = x + (1 + overshoot) * r_i
            x = np.clip(x, min_, max_)

            f = self.model.predict(x)
            gradient = self.model.gradient(x, pre_label)
            adv_label = np.argmax(f)
            logging.info('iteration={}, f[pre_label]={}, f[target_label]={}'
                         ', f[adv_label]={}, pre_label={}, adv_label={}'
                         ''.format(iteration, f[pre_label], (
                             f[adversary.target_label]
                             if adversary.is_targeted_attack else 'NaN'), f[
                                 adv_label], pre_label, adv_label))
            if adversary.try_accept_the_example(x, adv_label):
                return adversary

        return adversary
