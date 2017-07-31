"""Prepare THCHS-30 Chinese Speech Corpus.

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import os
import argparse
import soundfile
import json
from datasets.common import download, unpack

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_ROOT = "http://www.openslr.org/resources/18"
URL_CLEAN_DATA = URL_ROOT + "/data_thchs30.tgz"
URL_0DB_NOISY_TEST_DATA = URL_ROOT + "/test-noise.tgz"

MD5_CLEAN_DATA = "2d2252bde5c8429929e1841d4cb95e90"
MD5_0DB_NOISY_TEST_DATA = "7e8a985fb965b84141b68c68556c2030"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/THCHS30",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_prefix",
    default="manifest-thchs30",
    type=str,
    help="Filepath prefix for output manifests. (default: %(default)s)")
parser.add_argument(
    "--download_0db_noise_test",
    default="True",
    type=distutils.util.strtobool,
    help="Whether to download 0db noisy test dataset."
    " If True, download 0Db noise mixed test data. (default: %(default)s)")
parser.add_argument(
    "--remove_tar",
    default="True",
    type=distutils.util.strtobool,
    help="If True, remove tar file after unpacking automatically."
    " (default: %(default)s)")
parser.add_argument(
    "--char_transcription",
    default="True",
    type=distutils.util.strtobool,
    help="If True, transcription texts would be character-based "
    "and all whitespace in a transcription text would be removed. "
    "Otherwise transcription texts would be word-based."
    " (default: %(default)s)")
args = parser.parse_args()


def create_manifest(transcript_data_dir, audio_data_dir, manifest_path,
                    char_transcription):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path)
    json_lines = []
    filelist = sorted(os.walk(audio_data_dir))[0][2]
    audio_filelist = [fname for fname in filelist if fname.endswith('.wav')]
    for audio_file in audio_filelist:
        transcript_file_path = os.path.join(transcript_data_dir,
                                            audio_file + '.trn')
        if not os.path.isfile(transcript_file_path):
            raise IOError("Transcript file %s not exists." % \
                    transcript_file_path)
        transcript_text = open(transcript_file_path).readline().strip()
        if char_transcription == True:
            transcript_text = ''.join(transcript_text.split(' '))
        audio_file_path = os.path.join(audio_data_dir, audio_file)
        audio_data, samplerate = soundfile.read(audio_file_path)
        duration = float(len(audio_data)) / samplerate
        json_lines.append(
            json.dumps(
                {
                    'audio_filepath': audio_file_path,
                    'duration': duration,
                    'text': transcript_text
                },
                ensure_ascii=False))
    with open(manifest_path, 'w') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def prepare_dataset(target_dir, manifest_prefix, char_transcription,
                    download_noisy, rm_tar):
    def download_unpack(url, md5sum, download_dir, unpack_dir, rm_tar):
        if not os.path.exists(unpack_dir):
            filepath = download(url, md5sum, download_dir)
            unpack(filepath, unpack_dir, rm_tar)
        else:
            print("Skip downloading and unpacking. Data already exists in %s" %
                  unpack_dir)

    clean_dir = os.path.join(target_dir, "Clean")
    download_unpack(URL_CLEAN_DATA, MD5_CLEAN_DATA, target_dir, clean_dir,
                    rm_tar)
    # create [train-clean|dev-clean|test-clean] manifest file
    if char_transcription == True:
        transcription_type = 'char'
    else:
        transcription_type = 'word'

    base_dir = os.path.join(clean_dir, 'data_thchs30')
    transcript_data_dir = os.path.join(base_dir, 'data')
    for data_type in ['train', 'dev', 'test']:
        manifest_path = '%s.%s-%s-clean' % \
                (manifest_prefix, transcription_type, data_type)
        audio_data_dir = os.path.join(base_dir, data_type)
        create_manifest(transcript_data_dir, audio_data_dir, manifest_path,
                        char_transcription)

    if download_noisy == True:
        # create test-0db-noise-[cafe|car|white] manifest file
        noisy_test_dir = os.path.join(target_dir, "0DB-Noisy-Test")
        download_unpack(URL_0DB_NOISY_TEST_DATA, MD5_0DB_NOISY_TEST_DATA,
                        target_dir, noisy_test_dir, rm_tar)
        noisy_base_dir = os.path.join(noisy_test_dir, 'test-noise', '0db')
        for data_type in ['cafe', 'car', 'white']:
            manifest_path = '%s.%s-test-0db-noise-%s' % \
                    (manifest_prefix, transcription_type, data_type)
            audio_data_dir = os.path.join(noisy_base_dir, data_type)
            create_manifest(transcript_data_dir, audio_data_dir, manifest_path,
                            char_transcription)


def main():
    target_dir = args.target_dir
    manifest_prefix = args.manifest_prefix
    download_noisy = False
    if args.download_0db_noise_test == True:
        download_noisy = True
    rm_tar = False
    if args.remove_tar == True:
        rm_tar = True
    char_transcription = False
    if args.char_transcription == True:
        char_transcription = True
    prepare_dataset(target_dir, manifest_prefix, char_transcription,
                    download_noisy, rm_tar)


if __name__ == '__main__':
    main()
