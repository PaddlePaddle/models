from __future__ import print_function
import logging
import os
import re

from libspeech.audio import SpeechDLSegment
from .base import Dataset, DataPartition, TextCleaner

logger = logging.getLogger(__name__)


class WSJSet(Dataset):
    def __init__(self, config, kwargs, verbose=True, ce_mode=False):

        # All alignment data used in cross_entropy training
        # are supposed to be JSONDataset
        assert not ce_mode
        super(WSJSet, self).__init__(config, kwargs, verbose)
        self.audio_files = []

        if os.path.exists(self.pickle_file):
            self.load_from_pickle()
            return

        if os.path.exists(self.prev_set):
            ''' This dataset is not expected to change at all.
                So if on finding a pickle from the previous iteration, just
                need to adjust to the config params
            '''
            self.load_from_pickle(self.prev_set)
            self.set_config_params(config, kwargs)
            return

        data_partitions = {
            DataPartition.train: 'train_si284',
            DataPartition.dev: 'test_dev93',
            DataPartition.test: 'test_eval92'
        }

        mask = self.load_mask(config)

        for partition, file_prefix in data_partitions.items():
            location = os.path.join(self.location, file_prefix)
            summary_location = location + '.summary'
            if not os.path.exists(summary_location):
                self.audio_files = self._read_files(location)
                summaryf = open(summary_location, 'w')
                for audio in self.audio_files:
                    line = '{0}\t{1}\t{2}\n'.format(audio[0], audio[1],
                                                    audio[2])
                    summaryf.write(line)
                summaryf.close()

            # TODO sanjeev Include worker information
            logger.info('{} exists'.format(summary_location))
            with open(summary_location) as summary_file:
                lines = summary_file.readlines()
                for line in lines:
                    location, duration, text = line.split('\t')
                    duration = float(duration)
                    worker = 1

                    if mask and mask.get(location, 0) == 0:
                        continue

                    utt = self.create_utterance(
                        location,
                        duration,
                        text.strip(),
                        'WSJ',
                        partition,
                        worker=worker)
                    if not utt:
                        logger.warn("{} skipped.".format(location))
                        continue
                    self.worker_count[utt.worker] += 1
                    self.records[utt.key] = utt

        logger.info('num of items:' + str(len(self.records)))
        self._split_data_set()

    def _load_rewrites(self):
        file_name = os.path.join(
            os.path.dirname(__file__), 'wsj', "rewrites.txt")
        rewrites = {}
        with open(file_name, 'r') as fid:
            for l in fid.readlines():
                line = l.strip().split()
                if len(line) == 1:
                    rewrites[line[0]] = ''
                else:
                    rewrites[line[0]] = line[1]
        return rewrites

    def _apply_rewrites(self, text):
        """
        takes dict of lines and scrubs, returns
        list of scrubbed lines
        """
        rewrites = self._load_rewrites()
        for k, line in text.items():
            clean_line = []
            for word in line.split():
                if word in rewrites:
                    word = rewrites[word]
                clean_line.append(word.lower())
            _, text[k] = TextCleaner.clean(' '.join(clean_line))
        return text

    def get_text(self, text_file):
        with open(text_file, 'r') as fid:
            text_lines = [line.split() for line in fid.readlines()]
            text_lines = dict((line[0], ' '.join(line[1:]))
                              for line in text_lines)
            return self._apply_rewrites(text_lines)

    def _read_files(self, data_prefix):
        audio_files = []
        if not os.path.exists(data_prefix + ".txt"):
            logger.warn('WSJ path [', data_prefix, '] does not exist!!')
            return []

        text = self.get_text(data_prefix + ".txt")
        for line in open(data_prefix + '_sph.scp'):
            filename = re.search("^.{8} (.*)$", line)
            audio_file = re.sub(".wv1", ".wav", filename.group(1))
            key = line.split()[0]

            audio = SpeechDLSegment.from_wav_file(audio_file)
            assert audio.frame_rate == 16000
            audio_files.append([audio_file, audio.duration_seconds, text[key]])
        return audio_files
