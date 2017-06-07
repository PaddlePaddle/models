""" Walla noise block usage

    {
        "type": "walla_noise",
        "rate": // usage rate between 0.0 and 1.0
        "source": // should be one of "turk", "freesound" and "chime";
                  // this must be specified if the source is turk, as a
                  // different loader is used.
        "tags": // To select all available tags, do not include this field.
                //
                // Currently supported for the freesound are
                //
                //   "vehicle",
                //   "voice",
                //   "musical_instrument",
                //   "house",
                //   "office", (paper/document handling sounds; small dataset)
                //   "industrial",
                //   "noise",
                //   "ambient",
                //   "random_recording",
                //   "random_effects",
                //   "<UNK>" (<- audio with unlabeled tags)
                //
                // and for chime,
                //
                //   "bus",
                //   "caf",
                //   "ped",
                //   "str"
                //
        "noise_dir": // Directory containing the noise files
        "index_file": // Index of noises of interest in noise_dir
        "snr_min": // Minimum snr
        "snr_max": // Maximum snr
        "tag_distr": // A dictionary describing the noise type distribution;
                     // maps from a tag in "tags" to a probability mass.
                     // The masses are automatically normalized.
    }
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import os
from collections import defaultdict

from libspeech import utils
from libspeech.audio import SpeechDLSegment, MSEC
from . import base
from . import audio_database

TURK = "turk"
USE_AUDIO_DATABASE_SOURCES = frozenset(["freesound", "chime"])
HALF_NOISE_LENGTH_MIN_THRESHOLD = 3.0
FIND_NOISE_MAX_ATTEMPTS = 20

logger = logging.getLogger(__name__)


def _get_turk_noise_files(noise_dir, index_file):
    """ Creates a map from duration => a list of noise filenames

    Args:
        noise_dir (str): Directory of noise files which contains
            "noise-samples-list"
        index_file (str): Noise list

    Returns:
        noise_files (defaultdict): A map of bins to noise files.
            Each key is the duration, and the value is a list of noise
            files binned to this duration. Each bin is 2 secs.

    Note:
        * noise-samples-list should contain one line per noise (wav) file
            along with its duration in milliseconds
    """
    noise_files = defaultdict(list)
    if not os.path.exists(index_file):
        logger.error('No noise files were found at {}'.format(index_file))
        return noise_files
    num_noise_files = 0
    rounded_durations = list(range(0, 65, 2))
    with open(index_file, 'r') as fl:
        for line in fl:
            fname = os.path.join(noise_dir, line.strip().split()[0])
            duration = float(line.strip().split()[1]) / MSEC
            # bin the noise files into length bins rounded by 2 sec
            bin_id = utils.get_first_smaller(rounded_durations, duration)
            noise_files[bin_id].append(fname)
            num_noise_files += 1
    logger.info('Loaded {} turk noise files'.format(num_noise_files))
    return noise_files


class WallaNoiseModel(base.ModelInterface):
    """ Noise addition block

    Attributes:
        snr_min (func[int->float]): minimum signal-to-noise ratio
        snr_max (func[int->float]): maximum signal-to-noise ratio
        noise_dir (func[int->str]): root of where noise files are stored
        index_file (func[int->str]): index of noises of interest in noise_dir
        source (func[int->str]): select one from
            - turk
            - freesound
            - chime
            Note that this field is no longer required for the freesound
            and chime
        tags (func[int->list]): optional parameter for specifying what
            particular noises we want to add. See above for the available tags.
        tag_distr (func[int->dict]): optional noise distribution
    """

    def __init__(self,
                 snr_min,
                 snr_max,
                 noise_dir,
                 source,
                 allow_downsampling=None,
                 index_file=None,
                 tags=None,
                 tag_distr=None):
        # Define all required parameter maps here.
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_dir = noise_dir
        self.source = source

        # Optional parameters.
        if allow_downsampling is None:
            allow_downsampling = base.parse_parameter_from(False)
        if index_file is None:
            index_file = base.parse_parameter_from("")
        if tags is None:
            tags = base.parse_parameter_from(None)
        if tag_distr is None:
            tag_distr = base.parse_parameter_from(None)
        self.allow_downsampling = allow_downsampling
        self.index_file = index_file
        self.tags = tags
        self.tag_distr = tag_distr

        # When new noise sources are added, make sure to define the
        # associated bookkeeping variables here.
        self.turk_noise_files = []
        self.turk_noise_dir = None
        self.audio_index = audio_database.AudioIndex()

    def _init_data(self, iteration):
        """ Preloads stuff from disk in an attempt (e.g. list of files, etc)
        to make later loading faster. If the data configuration remains the
        same, this function does nothing.

        Args:
            iteration (int): current iteration
        """
        noise_dir = self.noise_dir(iteration)
        index_file = self.index_file(iteration)
        source = self.source(iteration)

        if not index_file:
            if source == TURK:
                index_file = os.path.join(noise_dir, 'noise-samples-list')
                logger.debug("index_file not provided; " + "defaulting to " +
                             index_file)
            # elif source == TODO_SUPPORT_NON_AUDIO_DATABASE_BASED_SOURCES:
            else:
                if source != "":
                    assert source in USE_AUDIO_DATABASE_SOURCES, \
                        "{} not supported by audio_database".format(source)
                index_file = os.path.join(noise_dir,
                                          "audio_index_commercial.txt")
                logger.debug("index_file not provided; " + "defaulting to " +
                             index_file)

        if source == TURK:
            if self.turk_noise_dir != noise_dir:
                self.turk_noise_dir = noise_dir
                self.turk_noise_files = _get_turk_noise_files(noise_dir,
                                                              index_file)
        # elif source == TODO_SUPPORT_NON_AUDIO_DATABASE_BASED_SOURCES:
        else:
            if source != "":
                assert source in USE_AUDIO_DATABASE_SOURCES, \
                    "{} not supported by audio_database".format(source)
            self.audio_index.refresh_records_from_index_file(
                self.noise_dir(iteration), index_file, self.tags(iteration))

    def transform_audio(self, audio, text, iteration, rng):
        """Adds walla noise

        Args:
            audio (SpeechDLSegment): Input audio
            iteration (int): current iteration
            rng (random.Random): RNG to use for augmentation.

        Returns:
            (SpeechDLSegment, str, int)
                - sound with walla noise
                - label
                - number of bytes read from disk

        """
        # This handles the cases where the data source or directories change.
        self._init_data(iteration)
        source = self.source(iteration)
        allow_downsampling = self.allow_downsampling(iteration)
        if source == TURK:
            audio, read_size = self._add_turk_noise(audio, iteration, rng,
                                                    allow_downsampling)
        # elif source == TODO_SUPPORT_NON_AUDIO_DATABASE_BASED_SOURCES:
        else:
            audio, read_size = self._add_noise(audio, iteration, rng,
                                               allow_downsampling)
        return audio, text, read_size

    def _sample_snr(self, iteration, rng):
        """ Returns a float sampled in [`self.snr_min`, `self.snr_max`]
        if both `self.snr_min` and `self.snr_max` are non-zero.
        """
        snr_min = self.snr_min(iteration)
        snr_max = self.snr_max(iteration)
        sampled_snr = rng.uniform(snr_min, snr_max)
        return sampled_snr

    def _add_turk_noise(self, audio, iteration, rng, allow_downsampling):
        """ Adds a turk noise to the input audio.

        Args:
            audio (SpeechDLSegment): input audio
            iteration (int): current iteration
            rng (random.Random): random number generator
            allow_downsampling (bool): indicates whether downsampling
                is allowed

        Returns:
            (SpeechDLSegment, int)
                - sound with turk noise added
                - number of bytes read from disk
        """
        read_size = 0
        if len(self.turk_noise_files) > 0:
            snr = self._sample_snr(iteration, rng)
            # Draw the noise file randomly from noise files that are
            # slightly longer than the utterance
            noise_bins = sorted(self.turk_noise_files.keys())
            # note some bins can be empty, so we can't just round up
            # to the nearest 2-sec interval
            rounded_duration = utils.get_first_larger(noise_bins,
                                                      audio.length_in_sec)
            noise_fname = \
                rng.sample(self.turk_noise_files[rounded_duration], 1)[0]
            noise = SpeechDLSegment.from_wav_file(noise_fname)
            logger.debug('noise_fname {}'.format(noise_fname))
            logger.debug('snr {}'.format(snr))
            read_size = len(noise) * 2
            # May throw exceptions, but this is caught by
            # AudioFeaturizer.get_audio_files.
            audio = audio.add_noise(
                noise, snr, rng=rng, allow_downsampling=allow_downsampling)
        return audio, read_size

    def _add_noise(self, audio, iteration, rng, allow_downsampling):
        """ Adds a noise indexed in audio_database.AudioIndex.

        Args:
            audio (SpeechDLSegment): input audio
            iteration (int): current iteration
            rng (random.Random): random number generator
            allow_downsampling (bool): indicates whether downsampling
                is allowed

        Returns:
            (SpeechDLSegment, int)
                - sound with turk noise added
                - number of bytes read from disk
        """
        read_size = 0
        tag_distr = self.tag_distr(iteration)
        if not self.audio_index.has_audio(tag_distr):
            if tag_distr is None:
                if not self.tags(iteration):
                    raise RuntimeError("The noise index does not have audio "
                                       "files to sample from.")
                else:
                    raise RuntimeError("The noise index does not have audio "
                                       "files of the given tags to sample "
                                       "from.")
            else:
                raise RuntimeError("The noise index does not have audio "
                                   "files to match the target noise "
                                   "distribution.")
        else:
            # Compute audio segment related statistics
            audio_duration = audio.length_in_sec

            # Sample relevant augmentation parameters.
            snr = self._sample_snr(iteration, rng)

            # Perhaps, we may not have a sufficiently long noise, so we need
            # to search iteratively.
            min_duration = audio_duration + 0.25
            for _ in range(FIND_NOISE_MAX_ATTEMPTS):
                logger.debug("attempting to find noise of length "
                             "at least {}".format(min_duration))

                success, record = \
                    self.audio_index.sample_audio(min_duration,
                                                  rng=rng,
                                                  distr=tag_distr)

                if success is True:
                    noise_duration, read_size, noise_fname = record

                    # Assert after logging so we know
                    # what caused augmentation to fail.
                    logger.debug("noise_fname {}".format(noise_fname))
                    logger.debug("snr {}".format(snr))
                    assert noise_duration >= min_duration
                    break

                # Decrease the desired minimum duration linearly.
                # If the value becomes smaller than some threshold,
                # we half the value instead.
                if min_duration > HALF_NOISE_LENGTH_MIN_THRESHOLD:
                    min_duration -= 2.0
                else:
                    min_duration *= 0.5

            if success is False:
                logger.info("Failed to find a noise file")
                return audio, 0

            diff_duration = audio_duration + 0.25 - noise_duration
            if diff_duration >= 0.0:
                # Here, the noise is shorter than the audio file, so
                # we pad with zeros to make sure the noise sound is applied
                # with a uniformly random shift.
                noise = SpeechDLSegment.from_wav_file(noise_fname)
                noise = noise.pad_silence(diff_duration, sides="both")
            else:
                # The noise clip is at least ~25 ms longer than the audio
                # segment here.
                diff_duration = int(noise_duration * audio.sample_rate) - \
                    int(audio_duration * audio.sample_rate) - \
                    int(0.02 * audio.sample_rate)
                start = float(rng.randint(0, diff_duration)) / \
                    audio.sample_rate
                finish = min(start + audio_duration + 0.2, noise_duration)
                noise = SpeechDLSegment.slice_from_wav_file(noise_fname, start,
                                                            finish)

            if len(noise) < len(audio):
                # This is to ensure that the noise clip is at least as
                # long as the audio segment.
                num_samples_to_pad = len(audio) - len(noise)
                # Padding this amount of silence on both ends ensures that
                # the placement of the noise clip is uniformly random.
                silence = SpeechDLSegment(
                    np.zeros(num_samples_to_pad), audio.sample_rate)
                noise = SpeechDLSegment.concatenate(silence, noise, silence)

            audio = audio.add_noise(
                noise, snr, rng=rng, allow_downsampling=allow_downsampling)

        return audio, read_size
