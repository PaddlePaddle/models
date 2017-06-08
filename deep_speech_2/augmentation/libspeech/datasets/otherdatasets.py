from __future__ import print_function
import glob
import logging
import os

from .base import Dataset
from ..audio import SpeechDLSegment

logger = logging.getLogger(__name__)


class FisherSet(Dataset):
    # TODO, awni: lot's of repeat code for SwbdSet and FisherSet,
    # find way to re-use
    def __init__(self, config, kwargs, verbose=True, ce_mode=False):

        # All alignment data used in cross_entropy training
        # are supposed to be JSONDataset
        assert not ce_mode
        super(FisherSet, self).__init__(config, kwargs, verbose)
        self.audio_files = []

        if os.path.exists(self.pickle_file):
            self.load_from_pickle()
            self.set_config_params(config, kwargs)
            return

        summary_location = self.location + '.summary'
        if not os.path.exists(summary_location):
            self.audio_files = self._read_files(self.location)
            summaryf = open(summary_location, 'w')
            for audio in self.audio_files:
                line = '{0}\t{1}\t{2}\n'.format(audio[0], audio[1], audio[2])
                summaryf.write(line)
            summaryf.close()

        mask = self.load_mask(config)

        # TODO sanjeev Include worker information and move summary reading and
        # writing to a Dataset
        logger.info(summary_location, ' exists')
        with open(summary_location) as summary_file:
            lines = summary_file.readlines()
            for line in lines:
                location, duration, text = line.split('\t')
                duration = float(duration)
                worker = 1  # TODO set this properly

                if mask and mask.get(location, 0) == 0:
                    continue

                utt = self.create_utterance(
                    location, duration, text.strip(), 'Fisher', worker=worker)
                if not utt:
                    logger.warn("{} skipped.".format(location))
                    continue
                self.worker_count[utt.worker] += 1
                self.records[utt.key] = utt

        self._split_data_set()

    def get_text(self, text_file):
        with open(text_file, 'r') as fid:
            text_lines = [line.split() for line in fid.readlines()]
            return dict((line[0], ' '.join(line[1:])) for line in text_lines)

    def _read_files(self, data_prefix):
        audio_files = []
        if not os.path.exists(os.path.join(data_prefix, "text")):
            logger.warn('Fisher path [', data_prefix, '] does not exist!!')
            return []

        text = self.get_text(os.path.join(data_prefix, "text"))
        with open(os.path.join(data_prefix, "filelist"), 'r') as fid:
            wav_files = [l.strip() for l in fid.readlines()]

        with open(os.path.join(data_prefix, "segments"), 'r') as fid:
            segments = dict((l.split()[0], l.split()[1:])
                            for l in fid.readlines())

        for audio_file in wav_files:
            key = audio_file.split("/")[-1].split(".")[0]
            _, s, e = segments[key]
            time = float(e) - float(s)

            audio_files.append([audio_file, time, text[key]])

        return audio_files


class SwbdSet(Dataset):
    def __init__(self, config, kwargs, verbose=True):

        super(SwbdSet, self).__init__(config, kwargs, verbose)
        self.audio_files = []

        if os.path.exists(self.pickle_file):
            self.load_from_pickle()
            self.set_config_params(config, kwargs)
            return

        summary_location = self.location + '.summary'
        if not os.path.exists(summary_location):
            self.audio_files = self._read_files(self.location)
            summaryf = open(summary_location, 'w')
            for audio in self.audio_files:
                line = '{0}\t{1}\t{2}\n'.format(audio[0], audio[1], audio[2])
                summaryf.write(line)
            summaryf.close()

        mask = self.load_mask(config)

        # TODO sanjeev Include worker information and move summary reading and
        # writing to a Dataset
        logger.info(summary_location, ' exists')
        with open(summary_location) as summary_file:
            lines = summary_file.readlines()
            for line in lines:
                location, duration, text = line.split('\t')
                duration = float(duration)
                worker = 1  # TODO set this properly

                if mask and mask.get(location, 0) == 0:
                    continue

                utt = self.create_utterance(
                    location, duration, text.strip(), 'Swbd', worker=worker)
                if not utt:
                    logger.warn("{} skipped.".format(location))
                    continue
                self.worker_count[utt.worker] += 1
                self.records[utt.key] = utt

        self._split_data_set()

    def get_text(self, text_file):
        with open(text_file, 'r') as fid:
            text_lines = [line.split() for line in fid.readlines()]
            return dict((line[0], ' '.join(line[1:])) for line in text_lines)

    def _read_files(self, data_prefix):
        audio_files = []
        if not os.path.exists(os.path.join(data_prefix, "text")):
            logger.warn('Swbd path [', data_prefix, '] does not exist!!')
            return []

        text = self.get_text(os.path.join(data_prefix, "text"))
        wav_files = glob.glob(os.path.join(data_prefix, "*/*.wav"))

        for audio_file in wav_files:
            key = audio_file.split("/")[-1].split(".")[0]
            audio = SpeechDLSegment.from_wav_file(audio_file)
            assert audio.frame_rate == 8000
            audio_files.append([audio_file, audio.duration_seconds, text[key]])
        return audio_files
