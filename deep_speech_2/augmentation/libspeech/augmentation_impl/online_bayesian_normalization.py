""" Online bayesian normalization usage

    {
        "type": "online_bayesian_normalization",
        "rate": // Usage rate between 0.0 and 1.0.
        "target_db": // Target RMS value in decibels
        "prior_db": // Prior RMS estimate in decibels
        "prior_samples": // Prior strength in number of samples
    }
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import base


class OnlineBayesianNormalizationModel(base.ModelInterface):
    """ 
    Instantiates an online bayesian normalization module.
    """

    def __init__(self,
                 target_db,
                 prior_db,
                 prior_samples,
                 startup_delay=base.parse_parameter_from(0.0)):
        """
        Args:
            :param target_db: Target RMS value in decibels
            :type target_db: func[int->scalar]
            :param prior_db: Prior RMS estimate in decibels
            :type prior_db: func[int->scalar]
            :param prior_samples: Prior strength in number of samples
            :type prior_samples: func[int->scalar]
            :param startup_delay: Start-up delay in seconds during
                which normalization statistics is accrued.
            :type starup_delay: func[int->scalar]
        """
        self.target_db = target_db
        self.prior_db = prior_db
        self.prior_samples = prior_samples
        self.startup_delay = startup_delay

    def transform_audio(self, audio, text, iteration, rng):
        """
        Normalizes the input audio using the online Bayesian approach.

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
                - normalized audio
                - label
                - number of bytes read from disk
        """
        read_size = 0
        target_db = self.target_db(iteration)
        prior_db = self.prior_db(iteration)
        prior_samples = self.prior_samples(iteration)
        startup_delay = self.startup_delay(iteration)
        audio = audio.normalize_online_bayesian(
            target_db, prior_db, prior_samples, startup_delay=startup_delay)
        return audio, text, read_size
