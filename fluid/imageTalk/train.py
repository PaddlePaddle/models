#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright(c) 2018 PaddlePaddle.  All rights reserved.
# Created on 2018
#
# Author:Lin_Bo
# Version 1.0
# filename: train.py

import os
import paddle
from paddle import fluid
import paddle.fluid as fluid
import paddle.v2 as paddle
import reader
from utils import logger, parse_train_cmd, build_dict, load_pkl, load_default_data
from utils import get_default_dict_path, get_default_img_feat_path
from network_conf import lstm_cell, lstm_net, lstm_prediction, lstm_main
from paddle.fluid.layers import lstm_unit


def train(train_features_path=None,
          test_features_path=None,
          word_dict_path=None,
          img2sent_dict_path=None,
          model_save_dir="models",
          use_cuda=False,
          learning_rate=0.001,
          num_passes=10):
    """
    train window_net model or sentence_net model

    :params train_features_path: path of training data, if this parameter
        is not specified, flickr30k-images will be used to run this example
    :type train_features_path: str
    :params test_features_path: path of testing data, if this parameter
        is not specified, flickr30k-images will be used to run this example
    :type test_features_path: str
    :params word_dict_path: path of word dictionary data, if this parameter
        is not specified, a default dictionary file will be used to run this example
    :type word_dict_path: str
    :params use_cuda: whether use the cuda
    :type use_cuda: bool
    :params num_pass: train pass number
    :type num_pass: int
    """

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    use_default_data = (train_features_path is None)

    if use_default_data:
        logger.info(("No training data are provided, "
                     "use flickr30k-images to train the model."))

        logger.info("downloading flickr30k-images ...")
        default_data_train_dir, default_data_test_dir, tar_token_filename = load_default_data(
        )

        logger.info("define default path ...")
        img2sent_dict_path, word_dict_path = get_default_dict_path()
        train_features_path, test_features_path = get_default_img_feat_path()

        logger.info("please wait to build the word dictionary ...")
        if word_dict_path is None or not os.path.exists(word_dict_path):
            logger.info(("word dictionary is not given, the dictionary "
                         "is automatically built from the training data."))

            # build the word dictionary to map the original string-typed
            # words into integer-typed index
            build_dict(
                tar_token_filename,
                img2sent_dict_path,
                word_dict_path,
                minCount=5)

    logger.info("the word dictionary path is %s" % word_dict_path)

    # get index info
    img2sent_dict = load_pkl(img2sent_dict_path)
    word_dict = load_pkl(word_dict_path)
    word_num = len(word_dict)
    logger.info("word number is : %d." % (word_num))

    # get train data reader
    train_reader = paddle.reader.shuffle(
        reader.train_reader(train_features_path, img2sent_dict, word_dict),
        buf_size=5120)

    # get test data reader
    if train_features_path is not None:
        # here, because training and testing data share a same format,
        # we still use the reader.train_reader to read the testing data.
        test_reader = reader.train_reader(test_features_path, img2sent_dict,
                                          word_dict)
    else:
        test_reader = None

    # get size of word dictionary
    dict_dim = len(word_dict) + 1
    logger.info("length of word dictionary is : %d." % (dict_dim))

    # define the image2lstm interface input layers
    hidden = fluid.layers.data(name="hidden", shape=[4096], dtype="float32")
    cell = fluid.layers.data(name="cell", shape=[4096], dtype="float32")
    pre_word = fluid.layers.data(name="pre_words", shape=[1], dtype="int64")
    label = fluid.layers.data(name="words", shape=[1], dtype="int64")

    # return the network result
    cost, acc, prediction, prev_hidden, prev_cell = lstm_main(
        word_dict_dim=dict_dim,
        prev_hidden=hidden,
        prev_cell=cell,
        lstm_him_dim=128,
        emb_dim=128,
        pre_word=pre_word,
        word=label)

    # create optimizer
    sgd_optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    sgd_optimizer.minimize(cost)

    # create trainer
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(
        feed_list=[label, pre_word, hidden, cell], place=place)

    # initialize training network
    exe.run(fluid.default_startup_program())
    prog = fluid.default_main_program()

    # begin training network
    for pass_id in range(num_passes):

        ## running the train data
        data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0
        for i, data_ in enumerate(train_reader()):

            img_feat, word_list = data_
            prev_hidden_, prev_cell_ = img_feat, img_feat
            for ii, word in enumerate(word_list):

                if ii == 0:
                    pre_words = word
                    data_lstm = [[word, pre_words, prev_hidden_, prev_cell_]]
                    avg_cost_np, avg_acc_np, prev_hidden_, prev_cell_ = exe.run(
                        prog,
                        feed=feeder.feed(data_lstm),
                        fetch_list=[cost, acc, prev_hidden, prev_cell])

                else:
                    pre_words = word_list[ii - 1]
                    data_lstm = [[word, pre_words, prev_hidden_, prev_cell_]]
                    avg_cost_np, avg_acc_np, prev_hidden_, prev_cell_ = exe.run(
                        prog,
                        feed=feeder.feed(data_lstm),
                        fetch_list=[cost, acc, prev_hidden, prev_cell])

            data_size = len(word_list)
            total_acc += data_size * avg_acc_np
            total_cost += data_size * avg_cost_np
            data_count += data_size

            if (i + 5) % 10 == 0:
                logger.info("pass_id: %d, batch %d, avg_acc: %f, avg_cost: %f" %
                            (pass_id, i + 1, total_acc / data_count,
                             total_cost / data_count))

        avg_cost = total_cost / data_count
        avg_acc = total_acc / data_count
        logger.info("Train result -- pass_id: %d,  avg_acc: %f, avg_cost: %f" %
                    (pass_id, avg_acc, avg_cost))

        ## running the test data
        if test_reader is not None:
            data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0

            for i, data_ in enumerate(train_reader()):

                img_feat, word_list = data_
                prev_hidden_, prev_cell_ = img_feat, img_feat
                for ii, word in enumerate(word_list):
                    if ii == 0:
                        pre_words = word
                        data_lstm = [[
                            word, pre_words, prev_hidden_, prev_cell_
                        ]]
                        avg_cost_np, avg_acc_np, prev_hidden_, prev_cell_ = exe.run(
                            prog,
                            feed=feeder.feed(data_lstm),
                            fetch_list=[cost, acc, prev_hidden, prev_cell])

                    else:
                        pre_words = word_list[ii - 1]
                        data_lstm = [[
                            word, pre_words, prev_hidden_, prev_cell_
                        ]]

                        avg_cost_np, avg_acc_np, prev_hidden_, prev_cell_ = exe.run(
                            prog,
                            feed=feeder.feed(data_lstm),
                            fetch_list=[cost, acc, prev_hidden, prev_cell])

                data_size = len(word_list)
                total_acc += data_size * avg_acc_np
                total_cost += data_size * avg_cost_np
                data_count += data_size
                print("test ..")

            avg_cost = total_cost / data_count
            avg_acc = total_acc / data_count
            logger.info("Test result -- pass_id: %d,  avg_acc: %f, avg_cost: %f"
                        % (pass_id, avg_acc, avg_cost))

        ## save inference model
        epoch_model = model_save_dir + "/" + "img2sentence_epoch" + str(pass_id
                                                                        % 5)
        logger.info("Saving inference model at %s" % (epoch_model))

        ##prediction is the topology return value
        ##if we use the prediction value as the infer result
        fluid.io.save_inference_model(epoch_model,
                                      ["hidden", "cell", "pre_words"],
                                      [prediction, prev_hidden, prev_cell], exe)

    logger.info("Training has finished.")


def main(args):

    train(
        train_features_path=args.train_features_path,
        test_features_path=args.test_features_path,
        word_dict_path=args.word_dict,
        img2sent_dict_path=args.img2sent_dict,
        use_cuda=args.use_cuda,
        num_passes=args.num_passes,
        model_save_dir=args.model_save_dir)


if __name__ == "__main__":

    args = parse_train_cmd()
    main(args)
