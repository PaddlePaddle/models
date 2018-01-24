"""
This module provide the attack method for FGSM's implement.
"""
from __future__ import division

import logging
from collections import Iterable

import numpy as np

from .base import Attack


class GradientSignAttack(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method").
    This is therefore called the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """

    def _apply(self, adversary, epsilons=1000):
        assert adversary is not None

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()

        if adversary.is_targeted_attack:
            gradient = self.model.gradient([(adversary.original,
                                             adversary.target_label)])
            gradient_sign = -np.sign(gradient) * (max_ - min_)
        else:
            gradient = self.model.gradient([(adversary.original,
                                             adversary.original_label)])
            gradient_sign = np.sign(gradient) * (max_ - min_)
        for epsilon in epsilons:
            adv_img = adversary.original + epsilon * gradient_sign
            adv_img = np.clip(adv_img, min_, max_)
            adv_label = np.argmax(self.model.predict([(adv_img, 0)]))
            logging.info('epsilon = {:.3f}, pre_label = {}, adv_label={}'.
                         format(epsilon, pre_label, adv_label))
            if adversary.try_accept_the_example(adv_img, adv_label):
                return adversary

        return adversary


FGSM = GradientSignAttack
