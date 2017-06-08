""" 
Volume Change usage example:
    {
        "type": "volume_change",
        "rate": // Usage rate between 0.0 and 1.0.
        "min_gain_dBFS": // Minimum gain in dBFS
        "max_gain_dBFS": // Maximum gain in dBFS
    }
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from . import base

DEFAULT_SEED = 3141592653


class VolumeChangeModel(base.ModelInterface):
    """ 
    Instantiates a volume change model.

    This is used for multi-loudness training of PCEN. See

    https://arxiv.org/pdf/1607.05666v1.pdf

    for more detail. In their experiments, a gain is drawn uniformly
    at random from -45 dBFS to -15 dBFS.
    """

    def __init__(self, min_gain_dBFS, max_gain_dBFS):
        """
        Args:
            :param min_gain_dBFS: Minimum gain in dBFS
            :type min_gain_dBFS: func[int->scalar]
            :param max_gain_dBFS: Maximum gain in dBFS
            :type max_gain_dBFS: func[int->scalar]
        """
        self.min_gain_dBFS = min_gain_dBFS
        self.max_gain_dBFS = max_gain_dBFS
        self.rng = random.Random(DEFAULT_SEED)

    def transform_audio(self, audio, text, iteration, rng, console=None):
        """ 
        Change audio loudness.

        Args:
            :param audio: input audio
            :type audio: SpeechDLSegment
            :param iteration: current iteration
            :type iteration: int
            :param text: audio transcription
            :type text: basestring
            :param rng: RNG to use for augmentation
            :type rng: random.Random
            :param console: if not None, output augmentation-
                related log data on failure.
            :type consold: logging.Logger
s
        Returns:
            (SpeechDLSegment, str, int)
                - normalized audio
                - label
                - number of bytes read from disk
        """
        read_size = 0
        min_gain_dBFS = self.min_gain_dBFS(iteration)
        max_gain_dBFS = self.max_gain_dBFS(iteration)
        gain = rng.uniform(min_gain_dBFS, max_gain_dBFS)
        audio = audio.apply_gain(gain)
        return audio, text, read_size
