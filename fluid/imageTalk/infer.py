#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright(c) 2018 PaddlePaddle.  All rights reserved.
# Created on 2018
#
# Author:Lin_Bo
# Version 1.0
# filename: infer.py
#

import os
import paddle.fluid as fluid
import reader
from utils import logger
from utils import reverse_dict
from utils import load_pkl
from utils import get_default_dict_path
from utils import get_default_img_feat_path

MAX_LEN = 20


def infer(test_reader, use_cuda=False, model_path=None):
    """
    inference function
    """
    if model_path is None or not os.path.exists(model_path):
        print(str(model_path) + " cannot be found")
        return
    # get the reverse dict
    #      and define the index of interest word in the window
    #            (mast the same as index of train )
    reverse_word_dict = reverse_dict(word_dict)

    # define the input layers
    hidden = fluid.layers.data(name="hidden", shape=[4096], dtype="float32")
    cell = fluid.layers.data(name="cell", shape=[4096], dtype="float32")
    pre_word = fluid.layers.data(name="pre_words", shape=[1], dtype="int64")

    # init paddlepaddle
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[hidden, cell, pre_word], place=place)
    inference_scope = fluid.core.Scope()

    ##
    start_word_id = word_dict["__start__"]
    end_word_id = word_dict["__end__"]

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        for data_ in test_reader():
            # get the words index and words in char format
            # words_index = [d[0] for d in data_]
            words = [reverse_word_dict[d] for d in data_[1]]

            img_feat, word_list = data_
            prev_hidden_, prev_cell_, prediction = img_feat, img_feat, start_word_id
            prediction_list = []
            ##
            # use the infer to predict
            for ii in range(MAX_LEN):
                if ii == 0:
                    # data_lstm = [(start_word_id, start_word_id, img_feat, img_feat)]
                    data_lstm = [[prev_hidden_, prev_cell_, start_word_id]]
                    prediction, prev_hidden, prev_cell = exe.run(
                        inference_program,
                        feed=feeder.feed(data_lstm),
                        fetch_list=fetch_targets,
                        return_numpy=True)

                    prediction = prediction[0].argmax()
                else:
                    # pre_words = word_list[ii - 1]
                    data_lstm = [[prev_hidden_, prev_cell_, prediction]]

                    prediction, prev_hidden, prev_cell = exe.run(
                        inference_program,
                        feed=feeder.feed(data_lstm),
                        fetch_list=fetch_targets,
                        return_numpy=True)
                    prediction = prediction[0].argmax()
                prediction_list.append(prediction)

                if prediction == end_word_id:
                    break

            prediction_tag = [reverse_word_dict[p] for p in prediction_list]

            prediction_words = " ".join(prediction_tag)
            source_words = " ".join(words)
            # print the result for compare
            print("%s\ns_POS = %s\np_POS = %s" %
                  ("-" * 40, source_words, prediction_words))


if __name__ == "__main__":

    ## get default file ...
    logger.info("running model infer ...")
    img2sent_dict_path, word_dict_path = get_default_dict_path()
    train_features_path, test_features_path = get_default_img_feat_path()

    print(img2sent_dict_path, word_dict_path)
    print(train_features_path, test_features_path)

    exit()
    logger.info(
        "test_features_path = %s\nword_dict_path = %s\nimg2sent_dict_path = %s\n"
        % (test_features_path, word_dict_path, img2sent_dict_path))

    ##get dictionary
    logger.info("loading dictionary")
    word_dict = load_pkl(word_dict_path)
    img2sent_dict = load_pkl(img2sent_dict_path)

    ## we use the reader.train_reader to read the testing data.
    logger.info("loading test reader")
    test_reader = reader.train_reader(test_features_path, img2sent_dict,
                                      word_dict)

    ## running model infer ...
    logger.info("running model infer ...")
    epoch_path = "./models/img2sentence_epoch0"
    infer(test_reader, use_cuda=False, model_path=epoch_path)
