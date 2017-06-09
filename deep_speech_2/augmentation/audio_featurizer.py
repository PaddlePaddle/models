from __future__ import division
from builtins import str
from time import clock

import numpy as np
import random
import os
import logging
import codecs
import json

from libspeech import utils
from libspeech import augmentation
from libspeech import features
from libspeech.audio import SpeechDLSegment
from libspeech.error import SampleRateError
from libspeech import net

# old int16-based pydub AudioSegment was normalized to 70db so we'll
# stick with that convention
TARGET_DB = 70.0
DTYPE = np.float32

logger = logging.getLogger(__name__)


class AudioFeaturizer(utils.Featurizer):
    """ 
    A subclass of utils.Featurizer that will take minibatches
    of audio data and convert them to features that can then be
    ingested by a speech-dl model.

    This class maintains all state related to creating and deleting features
    for the Model. It interacts with `EpochOrchestrator`, which invokes the
    methods of this class.

    Note: It is very important that the Model and `EpochFeatureManipulator`
    agree on the filenames of all features, keys, sizes, labels, and text
    files.

    """

    def __init__(self, audio_pool, write_dir, char_map, data_config, rank,
                 cross_entropy_mode, delay_label, attention_mode):
        """
        Args:
            :param audio_pool: A pool-like object that has a `submit` method.
            :type audio_pool: Pool-like
            :param write_dir: Filepath to directory where feature files
                should be written. In the case of a model run, this means
                the rank-qualified data directory that is returned by
                `data_dir_prep.get_rank_data_dir`.
            :type write_dir:  basestring
            :param char_map: CharMap used for this model.
            :type char_map: CharMap
            :param data_config: The `data_config` dict from a json file and which
                has been processed / modified by `process_dir()`
            :type data:  dict
            :param rank: Rank of this process.
            :type rank:  int
            :param attention_mode: if true, add end_of_seq label to the targets.
            :type attention_mode: bool
        """

        # if true, will load blank-inserted labels and cross-entropy training
        self.cross_entropy_mode = cross_entropy_mode
        self.delay_label = delay_label
        self.audio_pool = audio_pool
        self._write_dir = write_dir
        self.char_map = char_map
        self.rank = rank
        self.attention_mode = attention_mode
        self.epoch = 0
        self.iteration = 0

        # file templates.
        self.new_file_template = utils.FileTemplate(self.write_dir,
                                                    'process_data', '.json')
        """
        self.text_template = utils.FileTemplate(self.write_dir, 'text', '.txt')
        self.labels_template = utils.FileTemplate(self.write_dir, 'labels',
                                                  '.txt')
        self.sizes_template = utils.FileTemplate(self.write_dir, 'sizes',
                                                 '.txt')
        self.keys_template = utils.FileTemplate(self.write_dir, 'keys', '.txt')
        self.feats_template = utils.FileTemplate(self.write_dir, 'feats',
                                                 '.npy')
        """

        self.jitter = data_config['jitter']
        self.spec_params = data_config['spec_params']
        self.post_spec_params = data_config['post_spec_params']
        if isinstance(self.post_spec_params, dict):
            self.post_spec_params = [self.post_spec_params]
        self.sample_rate = data_config['sample_rate']
        self.spec_type = data_config['spec_type']
        self.step = data_config['step']
        self.allow_downsampling = data_config['allow_downsampling']
        self.allow_upsampling = data_config['allow_upsampling']
        self.use_online_normalization = data_config['use_online_normalization']
        self.use_ewma_normalization = data_config['use_ewma_normalization']
        self.use_global_normalization = data_config['use_global_normalization']
        self.online_normalization_parameters = \
            data_config['online_normalization_parameters']
        self.ewma_normalization_parameters = \
            data_config['ewma_normalization_parameters']

        self.aug_pipeline = None \
            if "augmentation_pipeline" not in data_config \
            else augmentation.parse_pipeline_from(
                data_config["augmentation_pipeline"])

    def set_epoch(self, epoch):
        """
        Sets the number of complete epochs we have trained thus far.

        Note:
            This is used to
                - update the augmentation hyperparameters based on the
                  schedule given in the json config file.
                - TODO Add below whenever epoch is used.

        Args:
            :param epoch: number of complete epochs we have gone through
            :type epoch: int
        """
        self.epoch = epoch

    def set_iteration(self, iteration):
        """
        Sets the current iteration.

        Args:
            :param iteration: iteration
            :type iteration: int
        """
        self.iteration = iteration

    @property
    def write_dir(self):
        """
        Returns the directory where featurized data should be written. 
        """
        return self._write_dir

    def clear_minibatch(self, minibatch_id):
        """
        Delete all features on disk corresponding to minibatch_id

        Args:
            :param minibatch_id: Minibatch identifier
            :type minibatch_id: int

        Note:
            This is one of two functions called by `Orchestrator`
                - featurize_minibatch writes creates the features to disk
                - clear_minibatch removes the features from disk
                    (called once Model consumes it)
        """
        for template in (self.text_template, self.labels_template,
                         self.sizes_template, self.keys_template,
                         self.feats_template):
            instantiated_file = template.instantiate(minibatch_id)
            if os.path.exists(instantiated_file):
                os.remove(instantiated_file)
        logger.info('Removed {}'.format(minibatch_id))

    def sample_shift(self, rng):
        """
        Sample a shift value using `self.jitter` and `self.step`.

        Returns:
            sampled_shift (int): a sampled value.
            rng (random.Random): RNG to use.
        """
        if not self.jitter:
            sampled_shift = 0
        else:
            sampled_shift = rng.randint(-1, 1) * (self.step // 2)

        return sampled_shift

    def sample_snr(self, rng):
        """
        Returns an int sampled in [`self.snr_min`, `self.snr_max`]
        if both `self.snr_min` and `self.snr_max` are non-zero.
        """
        if self.snr_min == self.snr_max == 0:
            sampled_snr = None
        else:
            sampled_snr = rng.uniform(self.snr_min, self.snr_max)
            sampled_snr = round(sampled_snr, 0)
        return sampled_snr

    def get_audio_files(self, fname, duration, add_noise, text, rng):
        """
        Reads in the wav file corresponding to fname, possibly resamples
        according to the config options and performs data augmentation.

        Args:
            :param fname: Absolute path to wav/pcm/sequence file
            :type fname: str
            :param duration: duration of the audio in seconds of fname
            :type duration: scalar
            :param add_noise: whether we're performing the data augmentation
            :type add_noise: boolean
            :param rng: random number generator
            :type rng: random.Random

        Returns:
            audio (AudioSegment): Audio + optional augmentation
            read_size (int): Number of bytes read from disk
        """
        start_read = clock()
        audio = SpeechDLSegment.from_file(fname)
        logger.debug('fname {}'.format(fname))
        read_size = 2 * len(audio)  # assume int16 (2 byte) samples in wav

        # Do dynamic resampling if allowed
        if audio.sample_rate > self.sample_rate and self.allow_downsampling:
            audio = audio.resample(self.sample_rate)
        if audio.sample_rate < self.sample_rate and self.allow_upsampling:
            audio = audio.resample(self.sample_rate)

        if audio.sample_rate != self.sample_rate:
            raise SampleRateError("Audio sample rate is incorrect. "
                                  "It is {} and should be {}. "
                                  "Maybe you want to allow upsampling/"
                                  "downsampling in your data config?"
                                  .format(audio.sample_rate, self.sample_rate))

        if add_noise and self.aug_pipeline is not None:
            audio, text, rsize = self.aug_pipeline.transform_audio(
                audio, text, self.iteration, rng)
            read_size += rsize
        read_time = clock() - start_read
        return audio, text, read_size, read_time

    def featurize_minibatch(self, minibatch_id, utterances):
        """
        Featurize all utterances in a minibatch.

        Args:
            :param minibatch_id: Minibatch identifier.
            :type minibatch_id: int
            :param utterances: Sequence of StrippedUtterance instances.
            :paramutterances: iterable
        """

        # this seed is rank, epoch, and minibatch_id dependent.
        # this means we don't need to save any RNG state during
        # slurm timeouts as we can easily re-create this seed.
        seed = '{0}{1}{2}'.format(self.rank, self.epoch, minibatch_id)
        rng = random.Random(seed)

        featurize_start = clock()
        total_read_time, total_spec_time, total_read_size = 0, 0, 0

        minibatch_futures = []
        """
        net.read_files_async([utterance.fname for utterance in utterances
                              if net.check_net_url(utterance.fname)])
        """
        for i, utterance in enumerate(utterances):
            try:
                data = self.get_audio_files(utterance.fname, utterance.duration,
                                            utterance.add_noise, utterance.text,
                                            rng)
                audio, text, read_size, read_time = data
                utterance = utterance._replace(text=text)
                utterances[i] = utterance
                # This variable name was too long to fit below and this
                # was the only way to make pep8 happy.
                online_norm_params = self.online_normalization_parameters
                ewma_norm_params = self.ewma_normalization_parameters
                # Up/downsampling is already performed in get_audio_files.
                future = self.audio_pool.submit(
                    process_utterance,
                    audio,
                    utterance,
                    self.char_map,
                    self.step,
                    self.spec_type,
                    self.spec_params,
                    self.post_spec_params,
                    self.sample_rate,
                    self.sample_shift(rng),
                    allow_downsampling=False,
                    allow_upsampling=False,
                    use_online_normalization=self.use_online_normalization,
                    online_normalization_parameters=online_norm_params,
                    use_ewma_normalization=self.use_ewma_normalization,
                    ewma_normalization_parameters=ewma_norm_params,
                    cross_entropy_mode=self.cross_entropy_mode,
                    delay_label=self.delay_label,
                    attention_mode=self.attention_mode,
                    use_global_normalization=self.use_global_normalization)
                minibatch_futures.append(future)
                total_read_time += read_time
                total_read_size += read_size
            except SampleRateError as e:
                logger.error('Failed to process audio file {}. ({})'
                             .format(utterance, e))

        bandwidth = utils.make_bandwidth_string(total_read_size,
                                                total_read_time)
        minibatch = [mbf.result() for mbf in minibatch_futures if mbf.result()]

        # Ocassionally feature extraction fails, minibatch size may not match
        num_failures = len(utterances) - len(minibatch)
        if num_failures > 0:
            notice = "Too many preprocess failures {}".format(num_failures)
            assert num_failures <= len(minibatch), notice
            minibatch.extend(minibatch[:num_failures])

        minibatch = sorted(minibatch, key=lambda x: (x[0].shape[1], x[1]))
        width = max(item[0].shape[1] for item in minibatch)
        total_spec_time = sum(item[4] for item in minibatch)
        write_time = self.write_minibatch(minibatch, minibatch_id)

        # log time spent.
        time_passed = clock() - featurize_start
        log_format = "minibatch id={} num={} max={}"
        log_format = log_format.format(minibatch_id, len(minibatch), width)
        timers = (total_read_time, bandwidth, total_spec_time, write_time,
                  time_passed)
        timing_format = "read=%.2f @%s spec=%.2f write=%.2f time=%.2f" % timers
        logger.info(log_format + " " + timing_format)

    def write_minibatch(self, minibatch, mb_id):
        """ Writes the featurized minibatch to disk.

        Args:
            :param minibatch: list of features as returned by `process_utterance`.
            :type minibatch: list     
            :param mb_id: Minibatch identifier.
            :type mb_id: int
        Returns:
            write_time (float): Time taken in seconds for this function call
        """

        feats_list = []
        width, total = 0, 0
        write_start = clock()

        # save data related to the minibatch data.
        text_file = self.text_template.instantiate(mb_id)
        labels_file = self.labels_template.instantiate(mb_id)
        sizes_file = self.sizes_template.instantiate(mb_id)
        keys_file = self.keys_template.instantiate(mb_id)

        with codecs.open(text_file, 'w', encoding='utf-8') as tf, \
            open(labels_file, 'w') as lf, open(sizes_file, 'w') as sf, \
                open(keys_file, 'w') as kf:
            for item in minibatch:
                feats, text, tokenized_labels, filename, _ = item
                assert feats.shape[1] >= width
                width = max(width, feats.shape[1])
                feats_list.append(feats)
                total += feats.shape[1]
                lf.write(' '.join(tokenized_labels) + '\n')
                sf.write(str(feats.shape[1]) + '\n')
                kf.write(filename + '\n')
                tf.write(text + '\n')

        # save the actual minibatch feature data.
        features_file = self.feats_template.instantiate(mb_id)
        with open(features_file, 'w') as ff:
            all_feats = np.hstack(feats_list)
            assert all_feats.shape[1] == total
            np.save(ff, np.asfortranarray(all_feats))
        write_time = clock() - write_start
        return write_time


def process_audio(audio_segment,
                  step,
                  spec_type,
                  spec_params,
                  post_spec_params,
                  shift,
                  sample_rate,
                  allow_downsampling=False,
                  allow_upsampling=False,
                  use_online_normalization=False,
                  online_normalization_parameters={},
                  use_ewma_normalization=False,
                  ewma_normalization_parameters={},
                  use_global_normalization=True):
    """Normalizes audio and calculates spectrogram.

    Args:
        :param audio_segment: Input audio data to be processed for
            feeding into Model
        :type  audio_segment: SpeechDLSegment
        :param step: hop size (ms) for calculating spectrogram
        :type step: scalar
        :param spec_params: dict of parameters specific to spectrogram type
        :type spec_params: dict
        :param post_spec_params: list of dict of parameters specific to
            spectrogram postprocessing
        :type post_spec_params: list
        :param shift: number of ms of raw audio skipped for jittering input
        :type shift: scalar
        :param sample_rate: Normalize the audio to be of this sample rate
        :type sample_rate: scalar
        param allow_downsampling: Whether to allow dynamic downsampling
            of audio to match target `sample_rate`.
        :type allow_downsampling: boolean
        :param allow_upsampling: Whether to allow dynamic upsampling
            of audio to match target `sample_rate`.
        :type allow_upsampling: bool
        :param use_online_normalization: Whether to use online normalization
            instead of global/non-causal normalization.
        :type use_online_normalization: bool
        :param online_normalization_parameters: Parameters for the online
            Bayesian normalization method
        :type online_normalization_parameters; dict
        :param use_ewma_normalization: Whether to use exponentially weighted
            moving average normalization instead of global/non-causal
            normalization.
        :type use_ewma_normalization: boolean
        :param ewma_normalization_parameters: Parameters for the exponentially
            weighted moving average normalization method
        :type ewma_normalization_parameters: dict
        :param use_global_normalization: Whether to use global normalization.
        :type use_global_normalization: bool

    Returns:
        feats (np.array): log-spectogram of input audio [freq, time]

    Raises:
        SampleRateError: If audio_segment has incorrect sample rate.
    """
    # Do dynamic resampling if allowed
    if audio_segment.sample_rate > sample_rate and allow_downsampling:
        audio_segment = audio_segment.resample(sample_rate)
    if audio_segment.sample_rate < sample_rate and allow_upsampling:
        audio_segment = audio_segment.resample(sample_rate)

    if audio_segment.sample_rate != sample_rate:
        raise SampleRateError("Audio sample rate is incorrect. "
                              "It is {} and should be {}. "
                              "Maybe you want to allow upsampling/"
                              "downsampling in your data config?"
                              .format(audio_segment.sample_rate, sample_rate))

    if use_online_normalization:
        audio_segment = audio_segment.normalize_online_bayesian(
            target_db=TARGET_DB, **online_normalization_parameters)
    elif use_ewma_normalization:
        audio_segment = audio_segment.normalize_ewma(
            target_db=TARGET_DB, **ewma_normalization_parameters)
    elif use_global_normalization:
        audio_segment = audio_segment.normalize(target_db=TARGET_DB)
    feats, _ = features.compute_specgram(
        audio_segment,
        step=step,
        shift=shift,
        spec_type=spec_type,
        **spec_params)
    for params_set in post_spec_params:
        if not params_set:
            continue
        feats = features.postprocess_spectrogram(feats, **params_set)
    feats = feats.astype(DTYPE)

    return feats


def process_utterance(audio,
                      utterance,
                      char_map,
                      step,
                      spec_type,
                      spec_params,
                      post_spec_params,
                      sample_rate,
                      shift,
                      allow_downsampling,
                      allow_upsampling,
                      use_online_normalization,
                      online_normalization_parameters,
                      use_ewma_normalization,
                      ewma_normalization_parameters,
                      cross_entropy_mode,
                      delay_label,
                      attention_mode,
                      use_global_normalization=True):
    """ Unpacks audio, computes spectrogram and tokenizes the labels of the
       utterance.

    Args:
        :param audio: SpeechDLSegment corresponding to utterance
        :type audio: SpeechDLSegment
        :param utterance: background utterance data
        :type utterance: StrippedUtterance
        :param char_map: map from n-gram-chargram => index within alphabet
        :type char_map: CharMap
        :param step: hop size (ms) for calculating spectrogram
        :type step: scalar
        :param window: window size (ms) for calculating spectrogram
        :type window: scalar
        :param max_freq: cut-off frequency for calculating spectrogram
        :type max_freq: scalar
        :param shift: number of ms of raw audio skipped for jittering input
        :type shift: scalar
        :param sample_rate: Normalize the audio to be of this sample rate
        :type sample_rate: scalar
        :param allow_downsampling: Whether to allow dynamic downsampling
            of audio to match target `sample_rate`.
        :type allow_downsampling: boolean
        :param allow_upsampling: Whether to allow dynamic upsampling
            of audio to match target `sample_rate`.
        :type allow_upsampling: bool
        :param use_online_normalization: Whether to use online normalization
            instead of global/non-causal normalization.
        :type use_online_normalization: bool
        :param online_normalization_parameters: Parameters for the online
            Bayesian normalization method
        :type online_normalization_parameters; dict
        :param use_ewma_normalization: Whether to use exponentially weighted
            moving average normalization instead of global/non-causal
            normalization.
        :type use_ewma_normalization: boolean
        :param ewma_normalization_parameters: Parameters for the exponentially
            weighted moving average normalization method
        :type ewma_normalization_parameters: dict
        :param spec_params: dict of parameters specific to spectrogram type
        :type spec_params: dict
        :param post_spec_params: list of dict of parameters specific to
            spectrogram postprocessing
        :type post_spec_params: list
        :param use_global_normalization: Whether to use global normalization.
        :type use_global_normalization: bool

    Returns:
        feats (np.array): log-spectrogram of audio [freq, time]
        text (str): raw transcript string
        tokenized_labels (list of ints): representation of `text` as a sequence
            of ints to index into the Model output
        filename (str): Path to wav/pcm/sequence_file
        spec_time (float): Time take in seconds for this function call

    Raises:
        SampleRateError: If utterance sample rate does not match target
            sample rate.
    """
    filename = utterance.fname
    text = utterance.text

    start_spec = clock()
    try:
        feats = process_audio(
            audio,
            step,
            spec_type,
            spec_params,
            post_spec_params,
            shift,
            sample_rate,
            allow_downsampling=allow_downsampling,
            allow_upsampling=allow_upsampling,
            use_online_normalization=use_online_normalization,
            online_normalization_parameters=online_normalization_parameters,
            use_ewma_normalization=use_ewma_normalization,
            ewma_normalization_parameters=ewma_normalization_parameters,
            use_global_normalization=use_global_normalization)
    except SampleRateError as e:
        raise SampleRateError("{} Filename: {}".format(e, utterance.fname))
    tokenized_labels = char_map.apply(text, cross_entropy_mode, delay_label,
                                      attention_mode)
    spec_time = clock() - start_spec
    return feats, text, tokenized_labels, filename, spec_time
