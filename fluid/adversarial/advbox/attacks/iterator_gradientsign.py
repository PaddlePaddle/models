"""
This module provide the attack method for Iterator FGSM's implement.
"""
from __future__ import division

import logging
from collections import Iterable

import numpy as np

from .base import Attack


class IteratorGradientSignAttack(Attack):
    """
    This attack was originally implemented by Alexey Kurakin(Google Brain).
    Paper link: https://arxiv.org/pdf/1607.02533.pdf
    """

    def _apply(self, adversary, epsilons=100, steps=10):
        """
        Apply the iterative gradient sign attack.
        Args:
            adversary(Adversary): The Adversary object.
            epsilons(list|tuple|int): The epsilon (input variation parameter).
            steps(int): The number of iterator steps.
        Return:
            adversary(Adversary): The Adversary object.
        """

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1 / steps, num=epsilons + 1)[1:]

        pre_label = adversary.original_label
        min_, max_ = self.model.bounds()

        for epsilon in epsilons:
            adv_img = adversary.original
            for _ in range(steps):
                if adversary.is_targeted_attack:
                    gradient = self.model.gradient(adversary.original,
                                                   adversary.target_label)
                    gradient_sign = -np.sign(gradient) * (max_ - min_)
                else:
                    gradient = self.model.gradient(adversary.original,
                                                   adversary.original_label)
                    gradient_sign = np.sign(gradient) * (max_ - min_)
                adv_img = adv_img + gradient_sign * epsilon
                adv_img = np.clip(adv_img, min_, max_)
                adv_label = np.argmax(self.model.predict(adv_img))
                logging.info('epsilon = {:.3f}, pre_label = {}, adv_label={}'.
                             format(epsilon, pre_label, adv_label))
                if adversary.try_accept_the_example(adv_img, adv_label):
                    return adversary

        return adversary


IFGSM = IteratorGradientSignAttack
