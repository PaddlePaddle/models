#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import gzip
import paddle.v2 as paddle
import argparse
import cPickle

from reader import Reader
from network_conf import DNNmodel
from utils import logger
import numpy as np
import math


def parse_args():
    """
    parse arguments
    :return:
    """
    parser = argparse.ArgumentParser(
        description="PaddlePaddle Youtube Recall Model Example")
    parser.add_argument(
        '--model_path', type=str, required=True, help="path of the model")
    parser.add_argument(
        '--feature_dict',
        type=str,
        required=True,
        help="path of feature_dict.pkl")
    return parser.parse_args()


def get_item_vec_from_softmax(nce_w, nce_b):
    """
    get item vectors from softmax parameter 
    """
    if nce_w is None or nce_b is None:
        return None
    vector = []
    total_items_num = nce_w.shape[0]
    if total_items_num != nce_b.shape[1]:
        return None
    dim_vector = nce_w.shape[1] + 1
    for i in range(0, total_items_num):
        vector.append([])
        vector[i].append(nce_b[0][i])
        for j in range(1, dim_vector):
            vector[i].append(nce_w[i][j - 1])
    return vector


def convt_simple_lsh(vector):
    """
    do simple lsh conversion
    """
    max_norm = 0
    num_of_vec = len(vector)
    for i in range(0, num_of_vec):
        norm = np.linalg.norm(vector[i])
        if norm > max_norm:
            max_norm = norm
    for i in range(0, num_of_vec):
        vector[i].append(
            math.sqrt(
                math.pow(max_norm, 2) - math.pow(np.linalg.norm(vector[i]), 2)))
    return vector


def item_vector():
    """
    get item vectors
    """
    args = parse_args()

    # check argument
    assert os.path.exists(
        args.model_path), 'The model_path path does not exist.'
    assert os.path.exists(
        args.feature_dict), 'The feature_dict path does not exist.'

    paddle.init(use_gpu=False, trainer_count=1)

    with open(args.feature_dict) as f:
        feature_dict = cPickle.load(f)

    # load the trained model.
    with gzip.open(args.model_path) as f:
        parameters = paddle.parameters.Parameters.from_tar(f)

    nid_dict = feature_dict['history_clicked_items']
    nid_to_word = dict((v, k) for k, v in nid_dict.items())

    nce_w = parameters.get("nce_w")
    nce_b = parameters.get("nce_b")
    item_vector = convt_simple_lsh(get_item_vec_from_softmax(nce_w, nce_b))
    for i in range(0, len(item_vector)):
        itemid = nid_to_word[i]
        print itemid + "\t" + ",".join(map(str, item_vector[i]))


if __name__ == "__main__":
    item_vector()
