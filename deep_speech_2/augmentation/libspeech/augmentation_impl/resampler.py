""" Resampler usage

    {
        "type": "resampler",
        "rate": // Usage rate between 0.0 and 1.0.
        "new_sample_rate": // New sample rate in Hz.
    }
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base


class ResamplerModel(base.ModelInterface):
    """ 
    Instantiates a resampler module.
    """

    def __init__(self, new_sample_rate):
        """
        Args:
            :param new_sample_rate: New sample rate in Hz
            :type new_sample_rate: func[int->scalar]
        """
        self.new_sample_rate = new_sample_rate

    def transform_audio(self, audio, text, iteration, rng):
        """ Resamples the input audio to the target sample rate.

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
                - resampled audio
                - label
                - number of bytes read from disk
        """
        read_size = 0
        new_sample_rate = self.new_sample_rate(iteration)
        audio = audio.resample(new_sample_rate)
        return audio, text, read_size
