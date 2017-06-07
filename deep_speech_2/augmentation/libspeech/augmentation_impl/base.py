from __future__ import absolute_import
from abc import ABCMeta, abstractmethod


class ModelInterface(object):
    """
    Base class for the augmentation model class. Perform an initialization
    below.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def transform_audio(self, audio, text, iteration, rng):
        """
        Adds various effects to the input audio. Ideally, this code
        should be re-entrant but the current architecture doesn't require
        this behavior.

        Args:
            :param audio: input audio
            :type audio: AudioSegment
            :param text: audio transcription
            :type text: basestring
            :param iteration: current iteration
            :type iteration: int
            :param rng:  RNG to use for augmentation.
            :param type: random.Random

        Returns:
            (AudioSegment, str, int)
                - transformed sound
                - transformed label
                - number of bytes read from disk
        """
        pass


class IdentityModel(ModelInterface):
    """
    Returns the raw audio data
    """

    def __init__(self):
        pass

    def transform_audio(self, audio, text, iteration, rng):
        return audio, text, 0


class AugmentationPipeline(ModelInterface):
    """
    Performs each model in the sequence with its respective rate.
    """

    def __init__(self, models, rates):
        '''
        Args:
            :param models: List of objects derived from base.ModelInterface
            :type models: list of objects
            :param rates: List of rate functions
            :type rates: list 
        '''
        self.models = models
        self.rates = rates
        assert len(models) == len(rates), \
            "There should be an equal number of models and rates."

    def transform_audio(self, audio, text, iteration, rng):
        """
        See the base class for interface definition.
        """
        read_size = 0
        for model, rate in zip(self.models, self.rates):
            if rng.uniform(0., 1.) <= rate(iteration):
                audio, text, rsize = model.transform_audio(audio, text,
                                                           iteration, rng)
                read_size += rsize
        return audio, text, read_size


def parse_parameter_from(config):
    '''
    TODO Define time-varying parameter specification and
    implement the configuration block parser.
    
    For now, they are just constants.
    '''
    return lambda iteration: config
