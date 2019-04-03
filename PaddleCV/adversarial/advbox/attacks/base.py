"""
The base model of the model.
"""
import logging
from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class Attack(object):
    """
    Abstract base class for adversarial attacks. `Attack` represent an
    adversarial attack which search an adversarial example. subclass should
    implement the _apply() method.

    Args:
        model(Model): an instance of the class advbox.base.Model.

    """
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    def __call__(self, adversary, **kwargs):
        """
        Generate the adversarial sample.

        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.
        """
        self._preprocess(adversary)
        return self._apply(adversary, **kwargs)

    @abstractmethod
    def _apply(self, adversary, **kwargs):
        """
        Search an adversarial example.

        Args:
        adversary(object): The adversary object.
        **kwargs: Other named arguments.
        """
        raise NotImplementedError

    def _preprocess(self, adversary):
        """
        Preprocess the adversary object.

        :param adversary: adversary
        :return: None
        """
        assert self.model.channel_axis() == adversary.original.ndim

        if adversary.original_label is None:
            adversary.original_label = np.argmax(
                self.model.predict(adversary.original))
        if adversary.is_targeted_attack and adversary.target_label is None:
            if adversary.target is None:
                raise ValueError(
                    'When adversary.is_targeted_attack is true, '
                    'adversary.target_label or adversary.target must be set.')
            else:
                adversary.target_label = np.argmax(
                    self.model.predict(adversary.target))

        logging.info('adversary:'
                     '\n         original_label: {}'
                     '\n         target_label: {}'
                     '\n         is_targeted_attack: {}'
                     ''.format(adversary.original_label, adversary.target_label,
                               adversary.is_targeted_attack))
