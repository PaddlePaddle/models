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
from utils import logger, parse_train_cmd, build_dict, load_dict, load_default_data
from network_conf import window_net, sentence_net


def train(topology,
          train_data_dir=None,
          test_data_dir=None,
          word_dict_path=None,
          label_dict_path=None,
          model_save_dir="models",
          use_cuda=False,
          window_size=5,
          learning_rate=0.001,
          batch_size=64,
          num_passes=10):
    """
    train window_net model or sentence_net model

    :params train_data_path: path of training data, if this parameter
        is not specified, Brown Corpus will be used to run this example
    :type train_data_path: str
    :params test_data_path: path of testing data, if this parameter
        is not specified, Brown Corpus will be used to run this example
    :type test_data_path: str
    :params word_dict_path: path of word dictionary data, if this parameter
        is not specified, a default dictionary file will be used to run this example
    :type word_dict_path: str
    :params label_dict_path: path of label dictionary data, if this parameter
        is not specified, a default dictionary file will be used to run this example
    :type label_dict_path: str
    :params use_cuda: whether use the cuda
    :type use_cuda: bool
    :params window_size: size of window width
    :type window_size: int
    :params num_pass: train pass number
    :type num_pass: int
    """

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    use_default_data = (train_data_dir is None)

    if use_default_data:
        logger.info(("No training data are provided, "
                     "use Brown corpus to train the model."))

        logger.info("downloading Brown corpus...")
        train_data_dir, test_data_dir, word_dict_path, label_dict_path = load_default_data(
        )

        logger.info("please wait to build the word dictionary ...")

    if word_dict_path is None or not os.path.exists(word_dict_path):
        logger.info(("word dictionary is not given, the dictionary "
                     "is automatically built from the training data."))

        # build the word dictionary to map the original string-typed
        # words into integer-typed index
        build_dict(
            data_dir=train_data_dir,
            save_path=word_dict_path,
            use_col=0,
            cutoff_fre=1,
            insert_extra_words=["<UNK>"])
    logger.info("the word dictionary path is %s" % word_dict_path)

    if not os.path.exists(label_dict_path):
        logger.info(("label dictionary is not given, the dictionary "
                     "is automatically built from the training data."))
        # build the label dictionary to map the original string-typed
        # label into integer-typed index
        build_dict(
            data_dir=train_data_dir,
            save_path=label_dict_path,
            use_col=1,
            cutoff_fre=10,
            insert_extra_words=["<UNK>"])
    logger.info("the label dictionary path is %s" % label_dict_path)

    # get index info
    word_dict = load_dict(word_dict_path)
    lbl_dict = load_dict(label_dict_path)
    class_num = len(lbl_dict)
    logger.info("class number is : %d." % (len(lbl_dict)))

    # get train data reader
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader.train_reader(train_data_dir, word_dict, lbl_dict,
                                window_size),
            buf_size=51200),
        batch_size=batch_size)

    # get test data reader
    if test_data_dir is not None:
        # here, because training and testing data share a same format,
        # we still use the reader.train_reader to read the testing data.
        test_reader = paddle.batch(
            reader.train_reader(test_data_dir, word_dict, lbl_dict,
                                window_size),
            batch_size=batch_size)
    else:
        test_reader = None

    # get size of word dictionary
    dict_dim = len(word_dict) + 1
    logger.info("length of word dictionary is : %d." % (dict_dim))

    # define the input layers
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    # return the network result
    cost, acc, prediction = topology(data, label, dict_dim, class_num=class_num)

    # create optimizer
    sgd_optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    sgd_optimizer.minimize(cost)

    # create trainer
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    # initialize training network
    exe.run(fluid.default_startup_program())
    prog = fluid.default_main_program()

    # begin training network
    for pass_id in range(num_passes):

        ## running the train data
        data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0
        for i, data_ in enumerate(train_reader()):
            avg_cost_np, avg_acc_np = exe.run(prog,
                                              feed=feeder.feed(data_),
                                              fetch_list=[cost, acc])
            data_size = len(data_)
            total_acc += data_size * avg_acc_np
            total_cost += data_size * avg_cost_np
            data_count += data_size

            if (i + 1) % 1000 == 0:
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
            for i, data in enumerate(test_reader()):
                avg_cost_np, avg_acc_np, prediction_np = exe.run(
                    prog,
                    feed=feeder.feed(data),
                    fetch_list=[cost, acc, prediction])
                data_size = len(data)
                total_acc += data_size * avg_acc_np
                total_cost += data_size * avg_cost_np
                data_count += data_size

            avg_cost = total_cost / data_count
            avg_acc = total_acc / data_count
            logger.info("Test result -- pass_id: %d,  avg_acc: %f, avg_cost: %f"
                        % (pass_id, avg_acc, avg_cost))

        ## save inference model
        epoch_model = model_save_dir + "/" + args.nn_type + "_epoch" + str(
            pass_id % 5)
        logger.info("Saving inference model at %s" % (epoch_model))

        ##prediction is the topology return value
        ##if we use the prediction value as the infer result
        fluid.io.save_inference_model(epoch_model, ["words"], prediction, exe)

    logger.info("Training has finished.")


def main(args):
    if args.nn_type == "window":
        topology = window_net
    elif args.nn_type == "sentence":
        topology = sentence_net

    train(
        topology=topology,
        train_data_dir=args.train_data_dir,
        test_data_dir=args.test_data_dir,
        word_dict_path=args.word_dict,
        label_dict_path=args.label_dict,
        use_cuda=args.use_cuda,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        model_save_dir=args.model_save_dir)


if __name__ == "__main__":

    args = parse_train_cmd()
    if args.train_data_dir is not None:
        assert args.word_dict and args.label_dict, (
            "the parameter train_data_dir, word_dict_path, and label_dict_path "
            "should be set at the same time.")
    main(args)
