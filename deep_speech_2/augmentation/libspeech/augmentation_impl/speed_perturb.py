"""
Speed perturbation module for making ASR robust to different voice
types (high pitched, low pitched, etc)
Samples uniformly between speed_min and speed_max
See reference paper here:
http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf
Speed perturbation usage
    {
        "type": "speed_perturb",
        "rate": // Usage rate between 0.0 and 1.0.
        "speed_min": // Minimum speed change (0.9 recommended)
        "speed_max": // Maximum speed change (1.1 recommended)
    }
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging

from . import base

logger = logging.getLogger(__name__)


class SpeedPerturbationModel(base.ModelInterface):
    """ 
    Instantiates a speed perturbation module.
    """

    def __init__(self, speed_min, speed_max):
        """
        Args:
            :param speed_min: Lower bound on new rate to sample
            :type speed_min: func[int->scalar]
            :param speed_max: Upper bound on new rate to sample
            :type speed_max: func[int->scalar]
        """
        if (speed_min < 0.9):
            logger.warn("Sampling speed below 0.9 can cause unnatural effects")
        if (speed_min > 1.1):
            logger.warn("Sampling speed above 1.1 can cause unnatural effects")
        self.speed_min = speed_min
        self.speed_max = speed_max

    def transform_audio(self, audio, text, iteration, rng):
        """ 
        Samples a new speed rate from the given range and
        changes the speed of the given audio clip.

        Args:
            :param audio: input audio
            :type audio: SpeechDLSegment
            :param iteration: current iteration
            :type iteration: int
            :param text: audio transcription
            :type text: basestring
            :param rng: RNG to use for augmentation
            :type rng: random.Random

        Returns:
            (SpeechDLSegment, text, int)
                - Speed-perturbed audio
                - label
                - number of bytes read from disk
        """
        read_size = 0
        speed_min = self.speed_min(iteration)
        speed_max = self.speed_max(iteration)
        sampled_speed = rng.uniform(speed_min, speed_max)
        audio = audio.speed_change(sampled_speed)
        return audio, text, read_size
