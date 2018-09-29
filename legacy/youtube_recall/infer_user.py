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


def infer_user():
    """
    infer_user
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

    nid_dict = feature_dict['history_clicked_items']
    nid_to_word = dict((v, k) for k, v in nid_dict.items())

    # load the trained model.
    with gzip.open(args.model_path) as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    parameters.set('_proj_province', \
            np.zeros(shape=parameters.get('_proj_province').shape))
    parameters.set('_proj_city', \
            np.zeros(shape=parameters.get('_proj_city').shape))
    parameters.set('_proj_phone', \
            np.zeros(shape=parameters.get('_proj_phone').shape))
    parameters.set('_proj_history_clicked_items', \
            np.zeros(shape= parameters.get('_proj_history_clicked_items').shape))
    parameters.set('_proj_history_clicked_categories', \
            np.zeros(shape= parameters.get('_proj_history_clicked_categories').shape))
    parameters.set('_proj_history_clicked_tags', \
            np.zeros(shape= parameters.get('_proj_history_clicked_tags').shape))

    # build model
    prediction_layer, fc = DNNmodel(
        dnn_layer_dims=[256, 31], feature_dict=feature_dict,
        is_infer=True).model_cost
    inferer = paddle.inference.Inference(
        output_layer=[prediction_layer, fc], parameters=parameters)

    reader = Reader(feature_dict)
    test_batch = []
    for idx, item in enumerate(
            reader.infer_user(['USER_ID_0', 'USER_ID_981', 'USER_ID_310806'])):
        test_batch.append(item)
    infer_a_batch(inferer, test_batch, nid_to_word)


def infer_a_batch(inferer, test_batch, nid_to_word):
    """
    input a batch of data and infer 
    """
    feeding = {
        'user_id': 0,
        'province': 1,
        'city': 2,
        'history_clicked_items': 3,
        'history_clicked_categories': 4,
        'history_clicked_tags': 5,
        'phone': 6
    }
    probs = inferer.infer(
        input=test_batch,
        feeding=feeding,
        field=["value"],
        flatten_result=False)
    for i, res in enumerate(zip(test_batch, probs[0], probs[1])):
        softmax_output = res[1]
        sort_nid = res[1].argsort()

        # print top 30 recommended item 
        ret = ""
        for j in range(1, 30):
            item_id = sort_nid[-1 * j]
            item_id_to_word = nid_to_word[item_id]
            ret += "%s:%.6f," \
                    % (item_id_to_word, softmax_output[item_id])
        print ret.rstrip(",")


if __name__ == "__main__":
    infer_user()
