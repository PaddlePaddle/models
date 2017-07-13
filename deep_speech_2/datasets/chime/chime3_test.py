"""Prepare CHiME3 test data.

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
import wget
import zipfile
import argparse
import soundfile
import json
from paddle.v2.dataset.common import md5file

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset/speech')

URL_SIMU = "https://d3s.myairbridge.com/filev2/86K57IC2DT69ZB4CF39U/?dlid=QMSR6W7G6EJ90T9HUQZWFK"
MD5_SIMU = "5d26dd3ef93cc5b4e498020b60940771"
FILENAME_SIMU = "CHiME3_isolated_et05_simu.zip"

URL_REAL = "https://d3s.myairbridge.com/packagev2/FDXXE174Y3RM9Q25/?dlid=1LG30HAZZOLXEOOAFWDYNU"
MD5_REAL = "bb944297966d49b30f7cf08aa2a4ae11"
FILENAME_REAL = "myairbridge-FDXXE174Y3RM9Q25.zip"

URL_TEXT = "https://d3s.myairbridge.com/filev2/3WATICK9YJV2B7B56K9T/?dlid=BFQ5YIGO2CQQC4W8XXQKKQ"
MD5_TEXT = "401eea064e4851aa56fa3b8fd78ab50a"
FILENAME_TEXT = "CHiME4_core.zip"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--target_dir",
    default=DATA_HOME + "/chime3_test",
    type=str,
    help="Directory to save the dataset. (default: %(default)s)")
parser.add_argument(
    "--manifest_filepath",
    default="manifest.chime3.test",
    type=str,
    help="Filepath for output manifests. (default: %(default)s)")
args = parser.parse_args()


def download(url, md5sum, target_dir, filename=None):
    """Download file from url to target_dir, and check md5sum."""
    if filename == None:
        filename = url.split("/")[-1]
    if not os.path.exists(target_dir): os.makedirs(target_dir)
    filepath = os.path.join(target_dir, filename)
    if not (os.path.exists(filepath) and md5file(filepath) == md5sum):
        print("Downloading %s ..." % url)
        wget.download(url, target_dir)
        print("\nMD5 Chesksum %s ..." % filepath)
        if not md5file(filepath) == md5sum:
            raise RuntimeError("MD5 checksum failed.")
    else:
        print("File exists, skip downloading. (%s)" % filepath)
    return filepath


def unpack(filepath, target_dir):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    if filepath.endswith('.zip'):
        zip = zipfile.ZipFile(filepath, 'r')
        zip.extractall(target_dir)
        zip.close()
    elif filepath.endswith('.tar') or filepath.endswith('.tar.gz'):
        tar = zipfile.open(filepath)
        tar.extractall(target_dir)
        tar.close()
    else:
        raise ValueError("File format is not supported for unpacking.")


def create_manifest(data_dir, prefix, manifest_path):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path)
    transcript_filepath = os.path.join(
        data_dir, "CHiME3/data/transcriptions/et05_" + prefix + ".trn_all")
    transcript = {}
    for line in open(transcript_filepath, 'r'):
        pos = line.find(' ')
        key = line[:pos]
        text = line[pos + 1:].strip().lower().replace('.', '')
        transcript[key] = text
    json_lines = []
    for subfolder, _, filelist in sorted(os.walk(data_dir)):
        for filename in filelist:
            if filename.endswith('CH1.wav') and subfolder.endswith(prefix):
                key = filename[:filename.find('.')]
                filepath = os.path.join(data_dir, subfolder, filename)
                audio_data, samplerate = soundfile.read(filepath)
                duration = float(len(audio_data)) / samplerate
                json_lines.append(
                    json.dumps({
                        'audio_filepath': filepath,
                        'duration': duration,
                        'text': transcript[key]
                    }))
    with open(manifest_path, 'w') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def main():
    if not os.path.exists(os.path.join(args.target_dir, "CHiME3")):
        # download
        filepath_simu = download(URL_SIMU, MD5_SIMU, args.target_dir,
                                 FILENAME_SIMU)
        filepath_real = download(URL_REAL, MD5_REAL, args.target_dir,
                                 FILENAME_REAL)
        filepath_text = download(URL_TEXT, MD5_TEXT, args.target_dir,
                                 FILENAME_TEXT)
        # unpack
        unpack(filepath_simu, args.target_dir)
        unpack(filepath_real, args.target_dir)
        unpack(filepath_text, args.target_dir)
        unpack(
            os.path.join(args.target_dir, 'CHiME3_isolated_et05_real.zip'),
            args.target_dir)
        unpack(
            os.path.join(args.target_dir, 'CHiME3_isolated_et05_bth.zip'),
            args.target_dir)
    else:
        print("Skip downloading and unpacking. Data already exists in %s." %
              args.target_dir)
    # create manifest json file
    create_manifest(args.target_dir, "simu", args.manifest_filepath + ".sim")
    create_manifest(args.target_dir, "real", args.manifest_filepath + ".real")
    create_manifest(args.target_dir, "bth", args.manifest_filepath + ".clean")


if __name__ == '__main__':
    main()
