# Copyright(c) 2018 PaddlePaddle.  All rights reserved.
# Created on 2018
#
# Author:Lin_Bo
# Version 1.0
# filename: utils.py

import logging
import os
import argparse
from collections import defaultdict
import hashlib
import zipfile
import random

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def parse_train_cmd():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle part-of-speech tag example.")
    parser.add_argument(
        "--nn_type",
        type=str,
        help=("A flag that defines which type of network to use, "
              "available: [window, sentence]."),
        default="dnn")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=False,
        help=("The path of training dataset (default: None). If this parameter "
              "is not set, Brown corpus will be used."),
        default=None)
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=False,
        help=("The path of testing dataset (default: None). If this parameter "
              "is not set, Brown corpus will be used."),
        default=None)
    parser.add_argument(
        "--word_dict",
        type=str,
        required=False,
        help=("The path of word dictionary (default: None). If this parameter "
              "is not set, Brown corpus will be used. If this parameter "
              "is set, but the file does not exist, word dictionay "
              "will be built from the training data automatically."),
        default=None)
    parser.add_argument(
        "--label_dict",
        type=str,
        required=False,
        help=("The path of label dictionay (default: None).If this parameter "
              "is not set, Brown corpus will be used. If this parameter "
              "is set, but the file does not exist, word dictionay "
              "will be built from the training data automatically."),
        default=None)
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        help="Whether use the cuda or not.")
    parser.add_argument(
        "--window_size", type=int, default=5, help="The size of window width.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="The learning rate of train.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The number of training examples in one forward/backward pass.")
    parser.add_argument(
        "--num_passes",
        type=int,
        default=10,
        help="The number of passes to train the model.")
    parser.add_argument(
        "--model_save_dir",
        type=str,
        required=False,
        help=("The path to save the trained models."),
        default="models")
    return parser.parse_args()


def build_dict(data_dir,
               save_path,
               use_col=0,
               cutoff_fre=0,
               insert_extra_words=[]):

    values = defaultdict(int)

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)

        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r") as fdata:
            for i, line in enumerate(fdata):
                if len(line) < 2:
                    continue

                for item in line.strip().split():
                    try:
                        w = item.split("/")[use_col]
                    except:
                        continue
                    values[w] += 1

    with open(save_path, "w") as f:
        for w in insert_extra_words:
            f.write("%s\t-1\n" % (w))

        for v, count in sorted(
                values.iteritems(), key=lambda x: x[1], reverse=True):
            if count < cutoff_fre: break
            f.write("%s\t%d\n" % (v, count))


def load_dict(dict_path):
    return dict((line.strip().split("\t")[0], idx)
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def load_reverse_dict(dict_path):
    return dict((idx, line.strip().split("\t")[0])
                for idx, line in enumerate(open(dict_path, "r").readlines()))


def reverse_dict(word_dict):
    return dict(zip(word_dict.values(), word_dict.keys()))


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def load_default_data():
    default_data_dir = "./data/"

    if not os.path.exists(default_data_dir):
        os.makedirs(default_data_dir)

    zip_filename = default_data_dir + "brown.zip"
    default_data_dir_brown = zip_filename[:-4]
    default_data_train_dir = os.path.join(default_data_dir_brown, "train")
    default_data_test_dir = os.path.join(default_data_dir_brown, "test")
    default_word_dict_path = os.path.join(default_data_dir_brown,
                                          "default_word.dict")
    default_label_dict_path = os.path.join(default_data_dir_brown,
                                           "default_label.dict")

    if not os.path.exists(default_data_dir_brown):
        os.makedirs(default_data_dir_brown)

    data_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
    md5sum = "a0a8630959d3d937873b1265b0a05497"

    if (not (os.path.exists(zip_filename)) or md5file(zip_filename) != md5sum):
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        os.system("wget -O %s -c %s" % (zip_filename, data_url))

    if not (os.path.exists(default_data_train_dir)) or not (
            os.path.exists(default_data_test_dir)):

        os.makedirs(default_data_train_dir)
        os.makedirs(default_data_test_dir)

        f = zipfile.ZipFile(zip_filename, 'r')
        for filename in f.namelist():
            if not filename[-1].isdigit():
                continue

            if random.random() > 0.2:
                save_dir = default_data_train_dir
            else:
                save_dir = default_data_test_dir

            data = f.read(filename)
            with open(os.path.join(save_dir, filename.split("/")[-1]),
                      'w+b') as f_:
                f_.write(data)
        f.close()

    return default_data_train_dir, default_data_test_dir, default_word_dict_path, default_label_dict_path
