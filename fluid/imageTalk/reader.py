# Copyright(c) 2018 PaddlePaddle.  All rights reserved.
# Created on 2018
#
# Author:Lin_Bo
# Version 1.0
# filename: reader.py
import os
import scipy.io
import random
from utils import load_pkl


def train_reader(img_feats_path, img2sent_dict, word_dict):
    """
    Reader interface for training data

    :param img_feats_path: image features data path
    :type img_feats_path: str
    :param img2sent_dict: image to sentence list map directory
    :type img2sent_dict: Python dict
    :param word_dict: path of word dictionary
    :type word_dict: Python dict


    """

    def reader():

        features_struct = scipy.io.loadmat(img_feats_path)

        img_paths = features_struct["img_paths"]
        feats = features_struct["feats"]
        img_feat_dim, N = feats.shape
        for i in range(N):
            img_name = img_paths[i].split("/")[-1]
            if not img2sent_dict[img_name]:
                continue
            sentence = random.choice(img2sent_dict[img_name])
            sentId = [word_dict[w] for w in sentence if w in word_dict.keys()]

            feat = feats[:, i]
            yield feat, sentId

    return reader
