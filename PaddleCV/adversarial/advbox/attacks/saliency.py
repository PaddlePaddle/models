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
This module provide the attack method for JSMA's implement.
"""
from __future__ import division

import logging
import random
import numpy as np

from .base import Attack


class SaliencyMapAttack(Attack):
    """
    Implements the Saliency Map Attack.
    The Jacobian-based Saliency Map Approach (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """

    def _apply(self,
               adversary,
               max_iter=2000,
               fast=True,
               theta=0.1,
               max_perturbations_per_pixel=7):
        """
        Apply the JSMA attack.
        Args:
            adversary(Adversary): The Adversary object.
            max_iter(int): The max iterations.
            fast(bool): Whether evaluate the pixel influence on sum of residual classes.
            theta(float): Perturbation per pixel relative to [min, max] range.
            max_perturbations_per_pixel(int): The max count of perturbation per pixel.
        Return:
            adversary: The Adversary object.
        """
        assert adversary is not None

        if not adversary.is_targeted_attack or (adversary.target_label is None):
            target_labels = self._generate_random_target(
                adversary.original_label)
        else:
            target_labels = [adversary.target_label]

        for target in target_labels:
            original_image = adversary.original

            # the mask defines the search domain
            # each modified pixel with border value is set to zero in mask
            mask = np.ones_like(original_image)

            # count tracks how often each pixel was changed
            counts = np.zeros_like(original_image)

            labels = range(self.model.num_classes())
            adv_img = original_image.copy()
            min_, max_ = self.model.bounds()

            for step in range(max_iter):
                adv_img = np.clip(adv_img, min_, max_)
                adv_label = np.argmax(self.model.predict(adv_img))
                if adversary.try_accept_the_example(adv_img, adv_label):
                    return adversary

                # stop if mask is all zero
                if not any(mask.flatten()):
                    return adversary

                logging.info('step = {}, original_label = {}, adv_label={}'.
                             format(step, adversary.original_label, adv_label))

                # get pixel location with highest influence on class
                idx, p_sign = self._saliency_map(
                    adv_img, target, labels, mask, fast=fast)

                # apply perturbation
                adv_img[idx] += -p_sign * theta * (max_ - min_)

                # tracks number of updates for each pixel
                counts[idx] += 1

                # remove pixel from search domain if it hits the bound
                if adv_img[idx] <= min_ or adv_img[idx] >= max_:
                    mask[idx] = 0

                # remove pixel if it was changed too often
                if counts[idx] >= max_perturbations_per_pixel:
                    mask[idx] = 0

                adv_img = np.clip(adv_img, min_, max_)

    def _generate_random_target(self, original_label):
        """
        Draw random target labels all of which are different and not the original label.
        Args:
            original_label(int): Original label.
        Return:
            target_labels(list): random target labels
        """
        num_random_target = 1
        num_classes = self.model.num_classes()
        assert num_random_target <= num_classes - 1

        target_labels = random.sample(range(num_classes), num_random_target + 1)
        target_labels = [t for t in target_labels if t != original_label]
        target_labels = target_labels[:num_random_target]

        return target_labels

    def _saliency_map(self, image, target, labels, mask, fast=False):
        """
        Get pixel location with highest influence on class.
        Args:
            image(numpy.ndarray): Image with shape (height, width, channels).
            target(int): The target label.
            labels(int): The number of classes of the output label.
            mask(list): Each modified pixel with border value is set to zero in mask.
            fast(bool): Whether evaluate the pixel influence on sum of residual classes.
        Return:
            idx: The index of optimal pixel.
            pix_sign: The direction of perturbation
        """
        # pixel influence on target class
        alphas = self.model.gradient(image, target) * mask

        # pixel influence on sum of residual classes(don't evaluate if fast == True)
        if fast:
            betas = -np.ones_like(alphas)
        else:
            betas = np.sum([
                self.model.gradient(image, label) * mask - alphas
                for label in labels
            ], 0)

        # compute saliency map (take into account both pos. & neg. perturbations)
        sal_map = np.abs(alphas) * np.abs(betas) * np.sign(alphas * betas)

        # find optimal pixel & direction of perturbation
        idx = np.argmin(sal_map)
        idx = np.unravel_index(idx, mask.shape)
        pix_sign = np.sign(alphas)[idx]

        return idx, pix_sign


JSMA = SaliencyMapAttack
