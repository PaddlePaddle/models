#Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import sys
import time
import argparse
import unittest
import contextlib
import numpy as np

import paddle.fluid as fluid

import utils, metric, configs
import models

from pretrained_word2vec import Glove840B_300D

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument(
    '--model_name', type=str, default='cdssmNet', help="Which model to train")
parser.add_argument(
    '--config',
    type=str,
    default='cdssm_base',
    help="The global config setting")
parser.add_argument(
    '--enable_ce',
    action='store_true',
    help='If set, run the task with continuous evaluation logs.')
parser.add_argument('--epoch_num', type=int, help='Number of epoch')

DATA_DIR = os.path.join(os.path.expanduser('~'), '.cache/paddle/dataset')


def evaluate(epoch_id, exe, inference_program, dev_reader, test_reader,
             fetch_list, feeder, metric_type):
    """
    evaluate on test/dev dataset
    """

    def infer(test_reader):
        """
        do inference function
        """
        total_cost = 0.0
        total_count = 0
        preds, labels = [], []
        for data in test_reader():
            avg_cost, avg_acc, batch_prediction = exe.run(
                inference_program,
                feed=feeder.feed(data),
                fetch_list=fetch_list,
                return_numpy=True)
            total_cost += avg_cost * len(data)
            total_count += len(data)
            preds.append(batch_prediction)
            labels.append(np.asarray([x[-1] for x in data], dtype=np.int64))
        y_pred = np.concatenate(preds)
        y_label = np.concatenate(labels)

        metric_res = []
        for metric_name in metric_type:
            if metric_name == 'accuracy_with_threshold':
                metric_res.append((metric_name, metric.accuracy_with_threshold(
                    y_pred, y_label, threshold=0.3)))
            elif metric_name == 'accuracy':
                metric_res.append(
                    (metric_name, metric.accuracy(y_pred, y_label)))
            else:
                print("Unknown metric type: ", metric_name)
                exit()
        return total_cost / (total_count * 1.0), metric_res

    dev_cost, dev_metric_res = infer(dev_reader)
    print("[%s] epoch_id: %d, dev_cost: %f, " % (time.asctime(
        time.localtime(time.time())), epoch_id, dev_cost) + ', '.join(
            [str(x[0]) + ": " + str(x[1]) for x in dev_metric_res]))

    test_cost, test_metric_res = infer(test_reader)
    print("[%s] epoch_id: %d, test_cost: %f, " % (time.asctime(
        time.localtime(time.time())), epoch_id, test_cost) + ', '.join(
            [str(x[0]) + ": " + str(x[1]) for x in test_metric_res]))
    print("")


def train_and_evaluate(train_reader, dev_reader, test_reader, network,
                       optimizer, global_config, pretrained_word_embedding,
                       use_cuda, parallel):
    """
    train network
    """

    # define the net
    if global_config.use_lod_tensor:
        # automatic add batch dim
        q1 = fluid.layers.data(
            name="question1", shape=[1], dtype="int64", lod_level=1)
        q2 = fluid.layers.data(
            name="question2", shape=[1], dtype="int64", lod_level=1)
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        cost, acc, prediction = network(q1, q2, label)
    else:
        # shape: [batch_size, max_seq_len_in_batch, 1]
        q1 = fluid.layers.data(
            name="question1", shape=[-1, -1, 1], dtype="int64")
        q2 = fluid.layers.data(
            name="question2", shape=[-1, -1, 1], dtype="int64")
        # shape: [batch_size, max_seq_len_in_batch]
        mask1 = fluid.layers.data(name="mask1", shape=[-1, -1], dtype="float32")
        mask2 = fluid.layers.data(name="mask2", shape=[-1, -1], dtype="float32")
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        cost, acc, prediction = network(q1, q2, mask1, mask2, label)

    if parallel:
        # TODO: Paarallel Training
        print("Parallel Training is not supported for now.")
        sys.exit(1)

    #optimizer.minimize(cost)
    if use_cuda:
        print("Using GPU")
        place = fluid.CUDAPlace(0)
    else:
        print("Using CPU")
        place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    if global_config.use_lod_tensor:
        feeder = fluid.DataFeeder(feed_list=[q1, q2, label], place=place)
    else:
        feeder = fluid.DataFeeder(
            feed_list=[q1, q2, mask1, mask2, label], place=place)

    # only for ce
    args = parser.parse_args()
    if args.enable_ce:
        SEED = 102
        fluid.default_startup_program().random_seed = SEED
        fluid.default_main_program().random_seed = SEED

    # logging param info
    for param in fluid.default_main_program().global_block().all_parameters():
        print("param name: %s; param shape: %s" % (param.name, param.shape))

    # define inference_program
    inference_program = fluid.default_main_program().clone(for_test=True)

    optimizer.minimize(cost)

    exe.run(fluid.default_startup_program())

    # load emb from a numpy erray
    if pretrained_word_embedding is not None:
        print("loading pretrained word embedding to param")
        embedding_name = "emb.w"
        embedding_param = fluid.global_scope().find_var(
            embedding_name).get_tensor()
        embedding_param.set(pretrained_word_embedding, place)

    evaluate(
        -1,
        exe,
        inference_program,
        dev_reader,
        test_reader,
        fetch_list=[cost, acc, prediction],
        feeder=feeder,
        metric_type=global_config.metric_type)

    # start training
    total_time = 0.0
    print("[%s] Start Training" % time.asctime(time.localtime(time.time())))
    for epoch_id in range(global_config.epoch_num):

        data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0
        batch_id = 0
        epoch_begin_time = time.time()
        for data in train_reader():
            avg_cost_np, avg_acc_np = exe.run(fluid.default_main_program(),
                                              feed=feeder.feed(data),
                                              fetch_list=[cost, acc])
            data_size = len(data)
            total_acc += data_size * avg_acc_np[0]
            total_cost += data_size * avg_cost_np[0]
            data_count += data_size
            if batch_id % 100 == 0:
                print("[%s] epoch_id: %d, batch_id: %d, cost: %f, acc: %f" %
                      (time.asctime(time.localtime(time.time())), epoch_id,
                       batch_id, avg_cost_np, avg_acc_np))
            batch_id += 1
        avg_cost = total_cost / data_count
        avg_acc = total_acc / data_count
        epoch_end_time = time.time()
        total_time += epoch_end_time - epoch_begin_time

        print("")
        print(
            "[%s] epoch_id: %d, train_avg_cost: %f, train_avg_acc: %f, epoch_time_cost: %f"
            % (time.asctime(time.localtime(time.time())), epoch_id, avg_cost,
               avg_acc, time.time() - epoch_begin_time))

        # only for ce
        if epoch_id == global_config.epoch_num - 1 and args.enable_ce:
            #Note: The following logs are special for CE monitoring.
            #Other situations do not need to care about these logs.
            gpu_num = get_cards(args)
            print("kpis\teach_pass_duration_card%s\t%s" % \
                  (gpu_num, total_time / (global_config.epoch_num)))
            print("kpis\ttrain_avg_cost_card%s\t%s" % (gpu_num, avg_cost))
            print("kpis\ttrain_avg_acc_card%s\t%s" % (gpu_num, avg_acc))

        epoch_model = global_config.save_dirname + "/" + "epoch" + str(epoch_id)
        fluid.io.save_inference_model(
            epoch_model, ["question1", "question2", "label"], acc, exe)

        evaluate(
            epoch_id,
            exe,
            inference_program,
            dev_reader,
            test_reader,
            fetch_list=[cost, acc, prediction],
            feeder=feeder,
            metric_type=global_config.metric_type)


def main():
    """
    This function will parse argments, prepare data and prepare pretrained embedding
    """
    args = parser.parse_args()
    global_config = configs.__dict__[args.config]()

    if args.epoch_num != None:
        global_config.epoch_num = args.epoch_num

    print("net_name: ", args.model_name)
    net = models.__dict__[args.model_name](global_config)

    # get word_dict
    word_dict = utils.getDict(data_type="quora_question_pairs")

    # get reader
    train_reader, dev_reader, test_reader = utils.prepare_data(
        "quora_question_pairs",
        word_dict=word_dict,
        batch_size=global_config.batch_size,
        buf_size=800000,
        duplicate_data=global_config.duplicate_data,
        use_pad=(not global_config.use_lod_tensor))

    # load pretrained_word_embedding
    if global_config.use_pretrained_word_embedding:
        word2vec = Glove840B_300D(
            filepath=os.path.join(DATA_DIR, "glove.840B.300d.txt"),
            keys=set(word_dict.keys()))
        pretrained_word_embedding = utils.get_pretrained_word_embedding(
            word2vec=word2vec, word2id=word_dict, config=global_config)
        print("pretrained_word_embedding to be load:",
              pretrained_word_embedding)
    else:
        pretrained_word_embedding = None

    # define optimizer
    optimizer = utils.getOptimizer(global_config)

    # use cuda or not
    if not global_config.has_member('use_cuda'):
        if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ[
                'CUDA_VISIBLE_DEVICES'] != '':
            global_config.use_cuda = True
        else:
            global_config.use_cuda = False

    global_config.list_config()

    train_and_evaluate(
        train_reader,
        dev_reader,
        test_reader,
        net,
        optimizer,
        global_config,
        pretrained_word_embedding,
        use_cuda=global_config.use_cuda,
        parallel=False)


def get_cards(args):
    if args.enable_ce:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        num = len(cards.split(","))
        return num
    else:
        return args.num_devices


if __name__ == "__main__":
    main()
