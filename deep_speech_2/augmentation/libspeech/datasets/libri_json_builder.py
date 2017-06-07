from __future__ import print_function
import glob
import json
import logging
import os
from ..utils import TextCleaner
from ..audio import SpeechDLSegment

logger = logging.getLogger(__name__)


def get_set_text(datapath):
    """
    Returns dictionary key is "spk-chp-id" and
    value is "<transcription>"
    """
    text_files = glob.glob("%s/*/*/*.txt" % datapath)
    text = {}
    for text_file in text_files:
        with open(text_file, 'r') as fid:
            lines = [l.split() for l in fid]
            lines = dict((l[0], " ".join(l[1:])) for l in lines)
        text.update(lines)
    return text


def clean(text):
    text = text.lower()
    assert all([t in TextCleaner.allowed_chars for t in text]), \
        "Unknown character"
    return text


def get_wav_duration(wavfile):
    return SpeechDLSegment.from_wav_file(wavfile).length_in_sec


def get_set_audio(datapath):
    """
    Returns dictionary key is "spk-chp-id" and
    value is (<path-to-wav>, <duration (ms)>)
    """
    wav_files = glob.glob("%s/*/*/*.wav" % datapath)
    audio_dict = {}
    for wav in wav_files:
        key = wav.split("/")[-1].split(".")[0]
        duration = get_wav_duration(wav)
        audio_dict[key] = (wav, duration)
    return audio_dict


def write_jsons(filename, audio_dict, text_dict):
    def speaker(key):
        return int(key.split("-")[0])

    with open(filename, 'w') as fid:
        num_files = 0
        for key, (path, duration) in audio_dict.items():
            text = text_dict[key]
            audio_file = {
                'key': path,
                'duration': duration,
                'text': clean(text),
                'speaker': speaker(key)
            }
            json.dump(audio_file, fid)
            fid.write("\n")
            num_files += 1
            if num_files % 100 == 0:
                logger.info("Done with %d" % num_files)


if __name__ == "__main__":
    libri_base = ("/mnt/data/speech-english/vol0/orig_data/librispeech/"
                  "LibriSpeech")
    train_sets = ['train-clean-100', 'train-clean-360', 'train-other-500']

    audio_dict = {}
    text_dict = {}

    for train_set in train_sets:
        datapath = os.path.join(libri_base, train_set)
        audio_dict.update(get_set_audio(datapath))
        text_dict.update(get_set_text(datapath))

    write_jsons(os.path.join(libri_base, "train.json"), audio_dict, text_dict)
