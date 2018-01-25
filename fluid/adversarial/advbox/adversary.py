"""
Defines a class that contains the original object, the target and the
adversarial example.

"""


class Adversary(object):
    """
    Adversary contains the original object, the target and the adversarial
    example.
    """

    def __init__(self, original, original_label=None):
        """
        :param original: The original instance, such as an image.
        :param original_label: The original instance's label.
        """
        assert original is not None

        self.__original = original
        self.__original_label = original_label
        self.__target_label = None
        self.__target = None
        self.__is_targeted_attack = False
        self.__adversarial_example = None
        self.__adversarial_label = None

    def set_target(self, is_targeted_attack, target=None, target_label=None):
        """
        Set the target be targeted or untargeted.

        :param is_targeted_attack: bool
        :param target: The target.
        :param target_label: If is_targeted_attack is true and target_label is
                    None, self.target_label will be set by the Attack class.
                    If is_targeted_attack is false, target_label must be None.
        """
        assert (target_label is None) or is_targeted_attack
        self.__is_targeted_attack = is_targeted_attack
        self.__target_label = target_label
        self.__target = target
        if not is_targeted_attack:
            self.__target_label = None
            self.__target = None

    def set_original(self, original, original_label=None):
        """
        Reset the original.

        :param original: Original instance.
        :param original_label: Original instance's label.
        """
        if original != self.__original:
            self.__original = original
            self.__original_label = original_label
            self.__adversarial_example = None
        if original is None:
            self.__original_label = None

    def _is_successful(self, adversarial_label):
        """
        Is the adversarial_label is the expected adversarial label.
        
        :param adversarial_label: adversarial label.
        :return: bool
        """
        if self.__target_label is not None:
            return adversarial_label == self.__target_label
        else:
            return (adversarial_label is not None) and \
                   (adversarial_label != self.__original_label)

    def is_successful(self):
        """
        Has the adversarial example been found.

        :return: bool
        """
        return self._is_successful(self.__adversarial_label)

    def try_accept_the_example(self, adversarial_example, adversarial_label):
        """
        If adversarial_label the target label that we are finding.
        The adversarial_example and adversarial_label will be accepted and
        True will be returned.

        :return: bool
        """
        assert adversarial_example.shape == self.__original.shape
        ok = self._is_successful(adversarial_label)
        if ok:
            self.__adversarial_example = adversarial_example
            self.__adversarial_label = adversarial_label
        return ok

    def perturbation(self, multiplying_factor=1.0):
        """
        The perturbation that the adversarial_example is added.

        :param multiplying_factor: float.
        :return: The perturbation that is multiplied by multiplying_factor.
        """
        assert self.__original is not None
        assert self.__adversarial_example is not None
        return multiplying_factor * (
            self.__adversarial_example - self.__original)

    @property
    def is_targeted_attack(self):
        """
        :property: is_targeted_attack
        """
        return self.__is_targeted_attack

    @property
    def target_label(self):
        """
        :property: target_label
        """
        return self.__target_label

    @target_label.setter
    def target_label(self, label):
        """
        :property: target_label
        """
        self.__target_label = label

    @property
    def target(self):
        """
        :property: target
        """
        return self.__target

    @property
    def original(self):
        """
        :property: original
        """
        return self.__original

    @property
    def original_label(self):
        """
        :property: original
        """
        return self.__original_label

    @original_label.setter
    def original_label(self, label):
        """
        original_label setter
        """
        self.__original_label = label

    @property
    def adversarial_example(self):
        """
        :property: adversarial_example
        """
        return self.__adversarial_example

    @adversarial_example.setter
    def adversarial_example(self, example):
        """
        adversarial_example setter
        """
        self.__adversarial_example = example

    @property
    def adversarial_label(self):
        """
        :property: adversarial_label
        """
        return self.__adversarial_label

    @adversarial_label.setter
    def adversarial_label(self, label):
        """
        adversarial_label setter
        """
        self.__adversarial_label = label
