from __future__ import print_function
from __future__ import division
from future import standard_library

standard_library.install_aliases()
from builtins import next
from builtins import object
import collections
import hashlib
import json
import logging
import numpy as np
import os
import pickle as pickle
import random
from .. import net
from ..utils import TextCleaner, get_label_from_alignment

split_char = ':'
from char_map import CharMap, SPACE, BLANK

logger = logging.getLogger(__name__)


class DataPartition(object):
    """An enum indicating which set an utterance belongs to.

    Possible valid values are
    {'UNK', 'train', 'dev', 'dev_spk', 'dev_noise', 'test',
        'dev_train_holdout'}

    Note:
        All utterances default to an 'unknown' partition by default.
        In later stage of preprocessing these utterances get binned into
        other partitions
    """
    UNK = 0
    train = 1
    dev = 2
    dev_spk = 3
    dev_noise = 4
    test = 5  # Currently ignored, handle with issue#176
    dev_train_holdout = 6

    _names = [
        'UNK', 'train', 'dev', 'dev_spk', 'dev_noise', 'test',
        'dev_train_holdout'
    ]

    @classmethod
    def from_str(cls, stri):
        try:
            return cls._names.index(stri)
        except ValueError:
            assert not stri, '%s should one of partition types!' % stri
        return 0

    @classmethod
    def repr(cls, val):
        assert val < len(cls._names), 'Invalid Enum value'
        return cls._names[val]


Utterance = collections.namedtuple('Utterance', [
    'key', 'duration', 'worker', 'add_noise', 'noise_spec', 'text', 'tag',
    'partition', 'part_hint', 'noisy', 'location', 'orderhash'
])


class Dataset(object):
    """Base dataset class to describe a set of audio-transcript pairs
    """

    def __init__(self, specs, data_config, verbose=True):
        """
        Args:
            specs (dict): Configures how this dataset should be built
                location (str): Path on disk where to pick read the dataset
                    from. Could be a directory or a file with more information.
                    So refer the docstring of the inherited class for specifics
                split (Optional[boolean]): Splice out a part of this dataset to
                    construct a dev and dev_spk set. Defaults to false, and so
                    all un-partitioned utterances are treated as training data
                num_train_sample(Optional[scalar]): integer indicating number
                    of utterances to be kept for sample dataset after
                    partitioning. These will also be present in the
                    training dataset, and will give a better measure of how
                    much the model is fitting to the training data.
                    Defaults to 0.
                split_train (Optional[boolean]): Splice out parts of the
                    training data to create a holdout dataset not based on
                    speaker ID.
                    Defaults to False.
                    The holdout dataset will be used to compare the performance
                    of the model across epochs to ensure overfitting does not
                    occur.
                    The sample dataset will be used to compare the performance
                    across different models.
                num_train_holdout(Optional[scalar]): integer indicating number
                    of utterances to be kept for holdout dataset after
                    partitioning. It will be used to view performance over
                    datasets across models.
                    Ensure split_train is set as true in `specs`.
                    Defaults to 0.
                train_hours (Optional[scalar]): Sample a part of the dataset
                    so that you can build a smaller dataset on the fly.
                    Defaults to None, and include all data
                add_noise (Optional[boolean]): Mix with noise while training?
                    Defaults to True
                prev_iter (Optional[boolean]): DEPRECATED

            data_config (dict): Configuration common to all datasets being
                considered by preprocess
                dev_set_sample_rate(Optional[scalar]): \in [0-1] indicating
                    number of train/dev items in the datasets after
                    partitioning. Make sure "split" is set in `specs`
                num_speakers_in_hold_out (Optional[int]): Number of speakers
                    to be used when creating the dev_spk set.
                    Only speakers who have more than 100 and fewer than 500
                    utterances are considered when building the dev_spk set,
                    so if specify a non-zero number here and your dev_spk is
                    empty, this is one possible reason.
                save_dir (Optional[str]): DEPRECATED
        """

        self.location = ''
        self.num_speakers_in_hold_out = 0.
        self.dev_set_sample_rate = 0.
        self.save_dir = ''
        self.pickle_fname = ''
        self.pickle_file = ''
        self.prev_set = ''
        self.split = False
        self.split_train = False
        self.num_train_sample = 0
        self.num_train_holdout = 0
        self.train_hours = False
        self.add_noise = True
        self.verbose = verbose

        self.train_utterances = []
        self.new_train_utterances = []
        self.dev_utterances = []
        self.speaker_held_out = []
        self.noisy_held_out = []
        self.train_holdout_utterances = []
        self.train_sample_utterances = []

        self.records = {}
        self.train_workers = set()
        self.dev_workers = set()
        self.train_holdout_workers = set()
        self.train_sample_workers = set()
        self.held_out_speakers = set()
        self.audio_files = {}
        self.noise_samples = []

        self.noise_added = 0
        self.num_noisy_train_utterances = 0
        self.train_duration = 0
        self.dev_duration = 0
        self.noisy_train_duration = 0
        self.noisy_held_out_duration = 0
        self.speaker_held_out_duration = 0
        self.train_holdout_duration = 0
        self.train_sample_duration = 0
        self.worker_count = collections.defaultdict(int)

        self.charset = set()
        self.missing_char_freq = collections.defaultdict(lambda: 0)
        self.using_char_map = False

        self.rng = random.Random(0)
        self.set_config_params(specs, data_config)

    def load_mask(self, config):
        """ Check whether there is a mask file. If yes, return its content
        Args:
            config (dict): Configures how this dataset should be built
        Returns:
            a mask dictionary
        """
        mask = {}
        if 'mask' in config:
            try:
                with open(config['mask']) as fp:
                    for line in fp:
                        mask.update(json.loads(line))
            except IOError:
                mask = {}
                logger.error('errors in loading {}'.format(config['mask']))
                logger.info('working on full dataset!!!')

        return mask

    def create_utterance(self,
                         location,
                         duration,
                         text,
                         tag,
                         part_hint=DataPartition.UNK,
                         noisy=False,
                         worker='',
                         add_noise=True,
                         noise_spec=None):
        """ Return a constructed Utterance object.
        Inputs:
            location (str): Location of utterance.
            duration (float): Utterance duration.
            text (str): Transcription.
            tag (str): Tag of utterance.
            part_hint (Optional[DataPartition, DataPartition.UNK]):
                desired partition to group utterance into.
            noisy (Optional[boolean, False]): Is utterance already noisy?
            worker (Optional[str, None]): Speaker ID.
            add_noise (Optional[boolean, True]): Whether to add noise or not.
            noise_spec (Optional[str, None]): Location of noise spec file.

        Returns:
            Utterance object, or None if one can't be created due to char_map
                constraints.
        """
        location = location.strip()
        key = location.strip()
        # the text is going to be further processed once the data items
        # are grouped into different partitions.
        # If cross entropy training mode and the datum is in train partition,
        # Then just replace "<BLANK>" with "_", Don't strip it, that will casue
        # the number of timeframes not equal to the text length.
        # If not Cross entropy training or the datum is not in train partition,
        # the text is going to be replaced by text.strip(), as usual.
        text = text.replace(BLANK, "_")
        tag = tag.strip()
        orderhash = os.path.basename(location)

        chars = set(list(text)) - set("_")
        # don't include <BLANK> into charmap
        if self.using_char_map:
            if ' ' in chars:
                chars.remove(' ')
                chars.add(SPACE)
            remainder = chars - self.charset
            if len(remainder) > 0:
                # indicate that this utternace
                # was missing all the elements in remainder.
                for r in remainder:
                    self.missing_char_freq[r] += 1
                return None
        else:
            self.charset.update(chars)

        if key.find('seq_') == -1 and key.find('seqbin_') == -1:
            orderhash = hashlib.md5(orderhash).hexdigest()

        return Utterance(
            key=location,
            duration=duration,
            worker=worker,
            text=text,
            add_noise=add_noise,
            noise_spec=noise_spec,
            tag=tag,
            partition=DataPartition.UNK,
            part_hint=part_hint,
            location=location,
            orderhash=orderhash,
            noisy=noisy)

    def set_config_params(self, specs, data_config):
        """These fields are populated from the configuration and are
           overwritten when loading from the pickle. Use this method to reset
           it to the correct values.

        Args:
            Read Dataset.__init__.__doc__
        """
        self.location = specs['location']
        self.split = specs.get('split', False)
        self.split_train = specs.get('split_train', False)
        self.train_hours = specs.get('train_hours', 0.0)
        self.add_noise = specs.get('add_noise', True)
        self.num_speakers_in_hold_out = 0
        self.dev_set_sample_rate = 0
        self.num_train_holdout = 0
        self.num_train_sample = 0

        if self.split:
            self.num_speakers_in_hold_out = \
                data_config.get('num_speakers_in_hold_out', 0)
            self.dev_set_sample_rate = \
                data_config.get('dev_set_sample_rate', 0)

        if self.split_train:
            self.num_train_holdout = \
                specs.get('num_train_holdout', 0)
            self.num_train_sample = \
                specs.get('num_train_sample', 0)

        self.save_dir = data_config['job_save_dir']
        name = os.path.basename(self.location)
        self.pickle_fname = '%s.%s.pkl' % (self.__class__.__name__, name)
        self.pickle_file = os.path.join(self.save_dir, self.pickle_fname)
        if 'prev_iter' in specs:
            self.prev_set = os.path.join(specs['prev_iter'], self.pickle_fname)
        if data_config['char_map']:
            self.charset = set(CharMap(data_config['char_map']).keys())
            self.using_char_map = True

    def get_charset(self):
        return TextCleaner.allowed_chars

    def check_sanity(self, utterances, duration):
        keys = set([utt.key for utt in utterances])
        assert len(utterances) == len(keys)
        assert np.isclose(
            duration,
            sum([utt.duration for utt in utterances]),
            atol=1e-3,
            rtol=0)
        parts = set([utt.partition for utt in utterances])
        assert ((len(parts) == 1 and
                 not next(iter(parts)) == DataPartition.UNK) or len(parts) == 0)

    def get_files(self):
        """Returns a shallow copy of the datasets and prints out durations

        Notes:
            This is the main interface used by preprocess.py to retreive
            items in the dataset
        """
        all_train = self.train_utterances + self.new_train_utterances
        self.check_sanity(all_train, self.train_duration)
        self.check_sanity(self.dev_utterances, self.dev_duration)
        self.check_sanity(self.speaker_held_out, self.speaker_held_out_duration)
        self.check_sanity(self.noisy_held_out, self.noisy_held_out_duration)
        self.check_sanity(self.train_holdout_utterances,
                          self.train_holdout_duration)
        self.check_sanity(self.train_sample_utterances,
                          self.train_sample_duration)
        if self.verbose:
            logger.info(' ==== {0} ==== '.format(self.__class__.__name__))
            logger.info('Num workers is {0}'.format(len(self.worker_count)))
            logger.info(
                "Train: {0} utterances, {1} speakers {2:3.2f} hours".format(
                    len(self.train_utterances) + len(self.new_train_utterances),
                    len(self.train_workers), self.train_duration / 3600.))
            logger.info("Speaker: {0} utterances, {1} speakers, {2:3.2f} hours"
                        .format(
                            len(self.speaker_held_out),
                            len(self.held_out_speakers),
                            self.speaker_held_out_duration / 3600.))
            logger.info("Noise: {0} utterances, {1:3.2f} hours".format(
                len(self.noisy_held_out), self.noisy_held_out_duration / 3600.))
            logger.info("Dev: {0} utterances, {1} speakers, {2:3.2f} hours"
                        .format(
                            len(self.dev_utterances),
                            len(self.dev_workers), self.dev_duration / 3600.))
            logger.info("Holdout: {0} utterances, {1} speakers, {2:3.2f} hours"
                        .format(
                            len(self.train_holdout_utterances),
                            len(self.train_holdout_workers),
                            self.train_holdout_duration / 3600.))
            logger.info("Sample: {0} utterances, {1} speakers, {2:3.2f} hours"
                        .format(
                            len(self.train_sample_utterances),
                            len(self.train_sample_workers),
                            self.train_sample_duration / 3600.))

            if sum(self.missing_char_freq.values()) > 0:
                logger.warn("Utterances skipped due to char-map constraints:")
                for k, v in sorted(
                        self.missing_char_freq.items(),
                        key=lambda x: x[1],
                        reverse=True):
                    logger.info("{} -> {}".format(k.encode('utf-8'), v))

        # Return shallow copies of the list.
        datasets = {
            'train': self.train_utterances[:],
            'dev_spk': self.speaker_held_out[:],
            'dev_noise': self.noisy_held_out[:],
            'dev': self.dev_utterances[:],
            'new_train': self.new_train_utterances[:],
            'dev_train_holdout': self.train_holdout_utterances[:],
            'dev_train_sample': self.train_sample_utterances[:]
        }
        return datasets

    def pickle(self):
        with open(self.pickle_file, 'w') as pickle_file:
            pickle.dump(self, pickle_file)

    def load_from_pickle(self, pickle_file=None):
        pickle_file = pickle_file or self.pickle_file
        logger.info('loading from the pickle', pickle_file)

        with open(pickle_file, 'r') as picklef:
            tmp_dict = pickle.load(picklef)
            self.__dict__.update(tmp_dict.__dict__)

    def _split_data_set(self, ce_mode=False):
        """Assigns all non-partitioned utterances into train/dev/dev_spk sets

        Notes:
           Must be called by all dataset subclasses to make sure that all
           non-partitioned utterances are assigned to one of the data partition
        """
        if not self.held_out_speakers:
            low_contrib_speakers = [
                wrker for wrker in self.worker_count.keys()
                if self.worker_count[wrker] > 100 and self.worker_count[wrker] <
                500
            ]

            num_speakers = min(self.num_speakers_in_hold_out,
                               len(low_contrib_speakers))

            self.held_out_speakers = self.rng.sample(low_contrib_speakers,
                                                     num_speakers)

        new_train_sample, new_train_holdout = 0, 0
        if self.split_train:
            n_train_holdout = max(
                0, self.num_train_holdout - len(self.train_holdout_utterances))
            train_usable_keys = [
                key for key in self.records.keys()
                if (self.records[key].worker not in self.held_out_speakers
                    ) and ((self.records[key].part_hint == DataPartition.UNK) or
                           (self.records[key].part_hint == DataPartition.train))
            ]
            if n_train_holdout > len(train_usable_keys):
                err_string = ('Error: number of utterances to be held out '
                              'for holdout ({0}) exceeds the number of '
                              'train samples in this dataset ({1}) : in '
                              'location{2} \n')

                raise Exception(
                    err_string.format(n_train_holdout,
                                      len(train_usable_keys), self.location))
            self.rng.shuffle(train_usable_keys)

            for key in train_usable_keys[0:n_train_holdout]:
                utterance = self.records[key]
                if utterance.worker in self.held_out_speakers:
                    continue
                if ce_mode:
                    txt = get_label_from_alignment(utterance.text)
                else:
                    txt = utterance.text
                utterance = utterance._replace(
                    partition=DataPartition.dev_train_holdout)
                utterance = utterance._replace(text=txt.strip())
                new_train_holdout += 1
                self.train_holdout_utterances.append(utterance)
                self.train_holdout_duration += utterance.duration
                self.train_holdout_workers.add(utterance.worker)
                self.records[key] = utterance

        add_dev = False if len(self.dev_utterances) > 0 else True
        add_noisy = False if len(self.noisy_held_out) > 0 else True

        new_train, new_noise, new_dev = 0, 0, 0
        for key in self.records.keys():
            utterance = self.records[key]

            if utterance.partition != DataPartition.UNK:
                continue

            rsample = self.rng.uniform(0.0, 1.0)
            # Speaker hold out
            if utterance.part_hint == DataPartition.dev_spk or \
                    (utterance.part_hint == DataPartition.UNK and
                     utterance.worker in self.held_out_speakers):
                utterance = utterance._replace(partition=DataPartition.dev_spk)
                utterance = utterance._replace(text=utterance.text.strip())
                self.speaker_held_out.append(utterance)
                self.speaker_held_out_duration += utterance.duration

            # Dev set
            elif utterance.part_hint == DataPartition.dev or \
                    (add_dev and self.dev_set_sample_rate >= rsample):
                new_dev += 1
                utterance = utterance._replace(partition=DataPartition.dev)
                utterance = utterance._replace(text=utterance.text.strip())
                self.dev_utterances.append(utterance)
                self.dev_workers.add(utterance.worker)
                self.dev_duration += utterance.duration

            # Noise dev set
            elif utterance.part_hint == DataPartition.dev_noise or \
                    (add_noisy and utterance.noisy and
                     3 * self.dev_set_sample_rate >= rsample):
                new_noise += 1
                utterance = \
                    utterance._replace(partition=DataPartition.dev_noise)
                utterance = utterance._replace(text=utterance.text.strip())
                self.noisy_held_out.append(utterance)
                self.noisy_held_out_duration += utterance.duration

            # Train set, Do not put test sets here!
            elif (utterance.part_hint == DataPartition.UNK or
                  utterance.part_hint == DataPartition.train):
                # Only allow self.train_hours of training data
                if self.train_hours and (
                        self.train_duration > self.train_hours * 3600):
                    continue

                new_train += 1
                utterance = utterance._replace(partition=DataPartition.train)
                # Add noise on a dataset by dataset basis
                # Should be done only for train
                # utterance.add_noise = utterance.add_noise and self.add_noise
                utterance = utterance._replace(add_noise=utterance.add_noise and
                                               self.add_noise)
                if not ce_mode:
                    utterance = utterance._replace(text=utterance.text.strip())
                # Add it to the train set, add noise is suggested
                self.new_train_utterances.append(utterance)
                # Noisy copies will also be marked as train

                # TODO Metrics are common for old and new training data
                self.train_duration += utterance.duration
                self.train_workers.add(utterance.worker)

            # update the utterance back to the record
            self.records[key] = utterance

        # now create the train_sample set.
        all_train_utterances = self.train_utterances \
            + self.new_train_utterances
        n_train_sample = max(
            0, self.num_train_sample - len(self.train_sample_utterances))
        if n_train_sample > len(all_train_utterances):
            err_string = ('Error: number of utterances to be used for '
                          'sample ({0}) exceeds the number of'
                          ' train utterances in this dataset ({1}) : '
                          'in location{2} \n')
            raise Exception(
                err_string.format(n_train_sample,
                                  len(all_train_utterances), self.location))
        if n_train_sample > 0:
            self.rng.shuffle(all_train_utterances)
            # Generating sample set. Do not consider utterances that will be
            # held out as a part of train_holdout or held_out_speakers
            for utterance in all_train_utterances[:n_train_sample]:
                new_train_sample += 1
                if ce_mode:
                    original_txt = get_label_from_alignment(utterance.text)
                    utterance = utterance._replace(text=original_txt.strip())
                self.train_sample_utterances.append(utterance)
                self.train_sample_duration += utterance.duration
                self.train_sample_workers.add(utterance.worker)

        if self.verbose:
            summary = "New {0} train, {1} dev, {2} noise," + \
                      " {3} train_holdout, {4} train_sample utterances"
            logger.info(
                summary.format(new_train, new_dev, new_noise, new_train_holdout,
                               new_train_sample))


class JSONDataset(Dataset):
    """ A generic dataset class read from a single file where each
        line specifies an utterance as a json object.

    A fully specified utterance looks like this:
        {"text": "yeah",
         "partition": "dev",
         "speaker": "Speaker1",
         "key": "/local/data/swbd/raw/sw02150-A/sw02150-A_038076-038225.wav",
         "duration": 1.49,
         "tag": "SwbdSet.raw"}

    Args:
        key (str): Pointer to the location of the wav/pcm/sequence files on
            (preferably local) disk
        text (str): Transcript of the audio
        duration (scalar): Duration of the audio in seconds. This is required
            because inferring this is expensive for large datasets and
            preprocessing requires this information apriori for
            mini-batching utterances of similar duration together
        partition (Optional[str]): string representation of DataPartition
        speaker (Optional[str]): Speaker Id if known, required if you need to
            build a dev_spk set from dataset
        tag (Optional[str]): Use this field to set tag level parameters,
            instead of dataset level parameters. We do not recommended using
            this field.
            This field is useful when you build a dataset comprised of several
            smaller datasets but you want to specify some parameters at the
            level of smaller datasets. Currently, add_noise is the only
            supported tag level parameter. Refer Dataset.__doc__ for dataset
            level parameters

    Notes:
        All the data-cleaning is expected to already be done, so if you leave
        special characters in the text field, it will become part of the
        alphabet.
        Char-set is recalculated, as the union of all chars in the text field.
    """

    def __init__(self, config, data_config, verbose=True, ce_mode=False):
        """
        Args:
            config (dict): Configures how this dataset should be built
                location (str): Path of data file on disk in JSONDataset format
                    Read JSONDataset.__doc__ for description of this format
                split (Optional[boolean]): Splice out a part of this dataset to
                    construct a dev and dev_spk set. Defaults to false, and so
                    all un-partitioned utternaces are treated as training data
                train_hours (Optional[scalar]): Sample a part of the dataset
                    so that you can build a smaller dataset on the fly.
                    Defaults to None, and include all data
                add_noise (Optional[boolean]): Mix with noise while training?
                    Defaults to True
                prev_iter (Optional[boolean]): DEPRECATED
                train_only (Optional[boolean]): If set, ignores all utterances
                    not marked as "train". Read JSONDataset.__doc__ for
                    information on how to mark an utterance for train-set.
                    defaults to False
                tag_specs (Optional[dict]):
                    tag_name => {'add_noise' => boolean}
                    You can override the add_noise specification of the dataset
                    to specify whether to add noise to specific slices of the
                    dataset or not.

            data_config (dict): Configuration common to all datasets being
                considered by preprocess
                dev_set_sample_rate(Optional[scalar]): \in [0-1] indicating
                    number of train/dev items in the datasets after
                    partitioning. Make sure "split" is set in `specs`
                num_speakers_in_hold_out (Optional[int]): Number of speakers
                    to be used when creating the dev_spk set.
                    Only speakers who have more than 100 and fewer than 500
                    utterances are considered when building the dev_spk set,
                    so if specify a non-zero number here and your dev_spk is
                    empty, this is one possible reason.
                save_dir (Optional[str]): DEPRECATED
        """
        super(JSONDataset, self).__init__(config, data_config, verbose)

        if os.path.exists(self.pickle_file):
            self.load_from_pickle()
            return

        if os.path.exists(self.prev_set):
            """ To keep things simple, we assume that this dataset is
                stationary. So if on finding a pickle from the previous
                iteration, load it, adjust the config params and return
            """
            self.load_from_pickle(self.prev_set)
            self.set_config_params(config, data_config)
            return

        self.train_only = config.get('train_only', False)
        mask = self.load_mask(config)

        for line in self.get_lines():
            try:
                spec = json.loads(line)
            except:
                logger.error("error reading this: %s" % line)

            key = spec['key'].strip()
            speaker = spec.get('speaker', '')
            duration = float(spec['duration'])
            # Temporary, issue logged issue#176
            partition = DataPartition.from_str(spec.get('partition', ''))
            tag = spec.get('tag', '')
            # JSONDataset has a train-only option so we can not include the
            # dev-set from a particular dataset
            if self.train_only and partition != DataPartition.train:
                continue

            if mask and mask.get(key, 0) == 0:
                continue

            add_noise = True
            if spec.get('tag') in config.get('tag_specs', {}):
                add_noise = config['tag_specs'][spec['tag']]['add_noise']

            utt = self.create_utterance(
                key,
                duration,
                spec['text'],
                tag,
                part_hint=partition,
                worker=speaker,
                add_noise=add_noise)
            if not utt:
                if self.verbose:
                    logger.warn("{} skipped.".format(spec['key']))
                continue
            self.worker_count[utt.worker] += 1
            self.records[utt.key] = utt

        self._split_data_set(ce_mode)

    def get_lines(self):
        # Well, get_lines() in NetJSONDataset will have the whole file in
        # memory before returning the files which makes yield less efficient
        # than return. Additionally using yield in JSONDataset will make
        # get_lines() inconsistent with NetJSONDataset.

        with open(self.location) as summary_file:
            lines = summary_file.readlines()
        return lines

    def get_charset(self):
        return self.charset


class NetJSONDataset(JSONDataset):
    def get_lines(self):
        net.read_file_async(self.location)
        desc_file = net.pop_object(self.location)
        lines = str.split(desc_file.strip(), "\n")
        logger.info("Read %d lines from %s" % (len(lines), self.location))
        return lines
