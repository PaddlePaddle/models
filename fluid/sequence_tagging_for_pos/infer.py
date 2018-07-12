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
import paddle
import paddle.fluid as fluid
import paddle.v2 as paddle
import reader
from utils import load_default_data
from utils import load_dict
from utils import logger
from utils import reverse_dict


def infer(test_reader, window_size=5, use_cuda=False, model_path=None):
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
    reverse_lbl_dict = reverse_dict(lbl_dict)
    interest_index = int(window_size / 2)

    # define the input layers
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    # init paddlepaddle
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data], place=place)
    inference_scope = fluid.core.Scope()

    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)
        for data_ in test_reader():
            # get the words index and words in char format
            words_index = [[d[0]] for d in data_]
            words = [reverse_word_dict[d[0][interest_index]] for d in data_]

            # use the infer to predict
            prediction = exe.run(inference_program,
                                 feed=feeder.feed(words_index),
                                 fetch_list=fetch_targets,
                                 return_numpy=True)

            # get the label tag and the prediction tag
            label_tag = [reverse_lbl_dict[d[1]] for d in data_]
            prediction_tag = [
                reverse_lbl_dict[p.argmax()] for p in prediction[0]
            ]

            # get the source string and prediction string of POS work
            source_POS = " ".join(
                ["/".join(items) for items in zip(words, label_tag)])
            prediction_POS = " ".join(
                ["/".join(items) for items in zip(words, prediction_tag)])

            # print the result for compare
            print("%s\ns_POS = %s\np_POS = %s" %
                  ("-" * 40, source_POS, prediction_POS))


if __name__ == "__main__":

    #define the test_data_dir, word_dict_path, label_dict_path
    train_data_dir, test_data_dir, word_dict_path, label_dict_path = load_default_data(
    )

    logger.info(
        "train_data_dir = %s\ntest_data_dir = %s\nword_dict_path = %s\nlabel_dict_path = %s\n"
        % (train_data_dir, test_data_dir, word_dict_path, label_dict_path))

    ##get dictionary
    logger.info("loading dictionary")
    word_dict = load_dict(word_dict_path)
    lbl_dict = load_dict(label_dict_path)

    ## we use the reader.train_reader to read the testing data.
    logger.info("loading test reader")
    test_reader = paddle.batch(
        reader.train_reader(test_data_dir, word_dict, lbl_dict), batch_size=32)

    ##running model infer ...
    logger.info("running model infer ...")
    epoch_path = "./models/window_epoch3"
    infer(test_reader, window_size=5, use_cuda=False, model_path=epoch_path)
