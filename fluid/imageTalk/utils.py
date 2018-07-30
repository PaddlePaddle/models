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
import tarfile
import random
import cPickle as pkl

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def parse_train_cmd():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle image to sentence example.")
    parser.add_argument(
        "--train_features_path",
        type=str,
        required=False,
        help=(
            "The path of training dataset (default: None). If this parameter "
            "is not set, flickr30k-images will be used."),
        default=None)
    parser.add_argument(
        "--test_features_path",
        type=str,
        required=False,
        help=("The path of testing dataset (default: None). If this parameter "
              "is not set, flickr30k-images will be used."),
        default=None)
    parser.add_argument(
        "--word_dict",
        type=str,
        required=False,
        help=("The path of word dictionary (default: None). If this parameter "
              "is not set, flickr30k-images will be used. If this parameter "
              "is set, but the file does not exist, word dictionay "
              "will be built from the training data automatically."),
        default=None)
    parser.add_argument(
        "--img2sent_dict",
        type=str,
        required=False,
        help=(
            "The path of img2sent dictionary (default: None). If this parameter "
            "is not set, flickr30k-images will be used. If this parameter "
            "is set, but the file does not exist, word dictionay "
            "will be built from the training data automatically."),
        default=None)
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        help="Whether use the cuda or not.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="The learning rate of train.")
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


def save_pkl(file_path, data):
    f = open(file_path, 'w')
    pkl.dump(data, f)
    f.close()


def load_pkl(file_path):
    with open(file_path, "r") as f_load:
        r_load = pkl.load(f_load)
    return r_load


def build_dict(tar_token_filename,
               default_img2sent_dict_path,
               default_word_dict_path,
               minCount=2):

    start_word, end_word = "__start__", "__end__"
    img2sent = defaultdict(list)
    wordDict = defaultdict(int)

    f = tarfile.open(tar_token_filename)
    for filename in f.getnames():
        if filename[-3:] == "txt":
            continue
        data = f.extractfile(filename).readlines()

        for line in data:
            items = line.strip().split("\t")
            if len(items) != 2:
                continue

            name = items[0].split("#")[0]

            sentence = [start_word] + items[1].split() + [end_word]
            img2sent[name].append(sentence)

            for word in sentence:
                wordDict[word] += 1

    save_pkl(default_img2sent_dict_path, img2sent)

    wordList = [items[0] for items in wordDict.items() if items[1] > minCount]
    wordDict = dict([(word, i) for i, word in enumerate(wordList)])
    save_pkl(default_word_dict_path, wordDict)

    f.close()


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

    tar_img_filename = default_data_dir + "flickr30k-images.tar"

    default_data_dir_flickr30k = tar_img_filename[:-4]
    default_data_train_dir = os.path.join(default_data_dir_flickr30k, "train")
    default_data_test_dir = os.path.join(default_data_dir_flickr30k, "test")

    tar_token_filename = default_data_dir + "flickr30k.tar.gz"

    if not os.path.exists(default_data_dir_flickr30k):
        os.makedirs(default_data_dir_flickr30k)

    img_url = "http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar"
    md5img = "618c6139b8fd2943e446b111f4ad9891"

    token_url = "http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz"
    md5token = "d3980ef9db5743acbd5bcd9394065067"

    if (not (os.path.exists(tar_img_filename)) or
            md5file(tar_img_filename) != md5img):
        if os.path.exists(tar_img_filename):
            os.remove(tar_img_filename)
        os.system("wget -O %s -c %s" % (tar_img_filename, img_url))

    if (not (os.path.exists(tar_token_filename)) or
            md5file(tar_token_filename) != md5token):
        if os.path.exists(tar_token_filename):
            os.remove(tar_token_filename)
        os.system("wget -O %s -c %s" % (tar_token_filename, token_url))

    if not (os.path.exists(default_data_train_dir)) or not (
            os.path.exists(default_data_test_dir)):

        os.makedirs(default_data_train_dir)
        os.makedirs(default_data_test_dir)

        f = tarfile.open(tar_img_filename)
        for filename in f.getnames():
            if filename[-3:] != "jpg":
                continue

            if random.random() > 0.2:
                save_dir = default_data_train_dir
            else:
                save_dir = default_data_test_dir

            data = f.extractfile(filename).read()
            with open(os.path.join(save_dir, filename.split("/")[-1]),
                      'w+b') as f_:
                f_.write(data)
        f.close()

    return default_data_train_dir, default_data_test_dir, tar_token_filename


def get_default_dict_path():
    default_data_dir = "./data/"
    default_data_dir_flickr30k = default_data_dir + "flickr30k-images"
    default_img2sent_dict_path = os.path.join(default_data_dir_flickr30k,
                                              "default_img2sentence.dict")
    default_word_dict_path = os.path.join(default_data_dir_flickr30k,
                                          "default_word.dict")
    return default_img2sent_dict_path, default_word_dict_path


def get_default_img_feat_path():
    default_data_dir = "./data/"
    default_img_feat_train_path = os.path.join(
        default_data_dir, "img_vgg_feats.dict")  #"./data/img_vgg_feats.dict"
    default_img_feat_test_path = os.path.join(
        default_data_dir,
        "img_vgg_feats_test.dict")  #"./data/img_vgg_feats_test.dict"

    return default_img_feat_train_path, default_img_feat_test_path


if __name__ == "__main__":

    default_data_train_dir, default_data_test_dir, tar_token_filename = load_default_data(
    )

    default_img2sent_dict_path, default_word_dict_path = get_default_dict_path(
    )

    build_dict(
        tar_token_filename,
        default_img2sent_dict_path,
        default_word_dict_path,
        minCount=5)
