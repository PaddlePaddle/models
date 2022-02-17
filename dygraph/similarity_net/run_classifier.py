#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
SimNet Task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import multiprocessing
import sys

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

sys.path.append("..")

import paddle
import paddle.fluid as fluid
import numpy as np
import config
import utils
import reader
import nets.paddle_layers as layers
import io
import logging

from utils import ArgConfig
from utils import load_dygraph
from model_check import check_version
from model_check import check_cuda


def train(conf_dict, args):
    """
    train process
    """

    # Get device
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    # run train
    logging.info("start train process ...")

    def valid_and_test(pred_list, process, mode):
        """
        return auc and acc
        """
        pred_list = np.vstack(pred_list)
        if mode == "test":
            label_list = process.get_test_label()
        elif mode == "valid":
            label_list = process.get_valid_label()
        if args.task_mode == "pairwise":
            pred_list = (pred_list + 1) / 2
            pred_list = np.hstack(
                (np.ones_like(pred_list) - pred_list, pred_list))
        metric.reset()
        metric.update(pred_list, label_list)
        auc = metric.eval()
        if args.compute_accuracy:
            acc = utils.get_accuracy(pred_list, label_list, args.task_mode,
                                     args.lamda)
            return auc, acc
        else:
            return auc

    with fluid.dygraph.guard(place):
        # used for continuous evaluation 
        if args.enable_ce:
            SEED = 102
            fluid.default_startup_program().random_seed = SEED
            fluid.default_main_program().random_seed = SEED

        # loading vocabulary
        vocab = utils.load_vocab(args.vocab_path)
        # get vocab size
        conf_dict['dict_size'] = len(vocab)
        conf_dict['seq_len'] = args.seq_len

        # Load network structure dynamically
        net = utils.import_class("./nets", conf_dict["net"]["module_name"],
                                 conf_dict["net"]["class_name"])(conf_dict)
        if args.init_checkpoint is not "":
            model, _ = load_dygraph(args.init_checkpoint)
            net.set_dict(model)
        # Load loss function dynamically
        loss = utils.import_class("./nets/losses",
                                  conf_dict["loss"]["module_name"],
                                  conf_dict["loss"]["class_name"])(conf_dict)
        # Load Optimization method
        learning_rate = conf_dict["optimizer"]["learning_rate"]
        optimizer_name = conf_dict["optimizer"]["class_name"]
        if optimizer_name == 'SGDOptimizer':
            optimizer = fluid.optimizer.SGDOptimizer(
                learning_rate, parameter_list=net.parameters())
        elif optimizer_name == 'AdamOptimizer':
            beta1 = conf_dict["optimizer"]["beta1"]
            beta2 = conf_dict["optimizer"]["beta2"]
            epsilon = conf_dict["optimizer"]["epsilon"]
            optimizer = fluid.optimizer.AdamOptimizer(
                learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                parameter_list=net.parameters())

        # load auc method
        metric = fluid.metrics.Auc(name="auc")
        simnet_process = reader.SimNetProcessor(args, vocab)

        # set global step
        global_step = 0
        ce_info = []
        losses = []
        start_time = time.time()

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=16,
            return_list=True,
            iterable=True,
            use_double_buffer=True)
        get_train_examples = simnet_process.get_reader(
            "train", epoch=args.epoch)
        train_loader.set_sample_list_generator(
            paddle.batch(
                get_train_examples, batch_size=args.batch_size), place)
        if args.do_valid:
            valid_loader = fluid.io.DataLoader.from_generator(
                capacity=16,
                return_list=True,
                iterable=True,
                use_double_buffer=True)
            get_valid_examples = simnet_process.get_reader("valid")
            valid_loader.set_sample_list_generator(
                paddle.batch(
                    get_valid_examples, batch_size=args.batch_size),
                place)
            pred_list = []

        if args.task_mode == "pairwise":

            for left, pos_right, neg_right in train_loader():

                left = fluid.layers.reshape(left, shape=[-1, 1])
                pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])
                neg_right = fluid.layers.reshape(neg_right, shape=[-1, 1])
                net.train()
                global_step += 1
                left_feat, pos_score = net(left, pos_right)
                pred = pos_score
                _, neg_score = net(left, neg_right)
                avg_cost = loss.compute(pos_score, neg_score)
                losses.append(np.mean(avg_cost.numpy()))
                avg_cost.backward()
                optimizer.minimize(avg_cost)
                net.clear_gradients()

                if args.do_valid and global_step % args.validation_steps == 0:
                    for left, pos_right in valid_loader():
                        left = fluid.layers.reshape(left, shape=[-1, 1])
                        pos_right = fluid.layers.reshape(
                            pos_right, shape=[-1, 1])
                        net.eval()
                        left_feat, pos_score = net(left, pos_right)
                        pred = pos_score

                        pred_list += list(pred.numpy())
                    valid_result = valid_and_test(pred_list, simnet_process,
                                                  "valid")
                    if args.compute_accuracy:
                        valid_auc, valid_acc = valid_result
                        logging.info(
                            "global_steps: %d, valid_auc: %f, valid_acc: %f, valid_loss: %f"
                            % (global_step, valid_auc, valid_acc,
                               np.mean(losses)))
                    else:
                        valid_auc = valid_result
                        logging.info(
                            "global_steps: %d, valid_auc: %f, valid_loss: %f" %
                            (global_step, valid_auc, np.mean(losses)))

                if global_step % args.save_steps == 0:
                    model_save_dir = os.path.join(args.output_dir,
                                                  conf_dict["model_path"])
                    model_path = os.path.join(model_save_dir, str(global_step))

                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    fluid.dygraph.save_dygraph(net.state_dict(), model_path)

                    logging.info("saving infer model in %s" % model_path)
        else:
            for left, right, label in train_loader():
                left = fluid.layers.reshape(left, shape=[-1, 1])
                right = fluid.layers.reshape(right, shape=[-1, 1])
                label = fluid.layers.reshape(label, shape=[-1, 1])
                net.train()
                global_step += 1
                left_feat, pred = net(left, right)
                avg_cost = loss.compute(pred, label)
                losses.append(np.mean(avg_cost.numpy()))
                avg_cost.backward()
                optimizer.minimize(avg_cost)
                net.clear_gradients()

                if args.do_valid and global_step % args.validation_steps == 0:
                    for left, right in valid_loader():
                        left = fluid.layers.reshape(left, shape=[-1, 1])
                        right = fluid.layers.reshape(right, shape=[-1, 1])
                        net.eval()
                        left_feat, pred = net(left, right)
                        pred_list += list(pred.numpy())
                    valid_result = valid_and_test(pred_list, simnet_process,
                                                  "valid")
                    if args.compute_accuracy:
                        valid_auc, valid_acc = valid_result
                        logging.info(
                            "global_steps: %d, valid_auc: %f, valid_acc: %f, valid_loss: %f"
                            % (global_step, valid_auc, valid_acc,
                               np.mean(losses)))
                    else:
                        valid_auc = valid_result
                        logging.info(
                            "global_steps: %d, valid_auc: %f, valid_loss: %f" %
                            (global_step, valid_auc, np.mean(losses)))

                if global_step % args.save_steps == 0:
                    model_save_dir = os.path.join(args.output_dir,
                                                  conf_dict["model_path"])
                    model_path = os.path.join(model_save_dir, str(global_step))

                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    fluid.dygraph.save_dygraph(net.state_dict(), model_path)

                    logging.info("saving infer model in %s" % model_path)

        end_time = time.time()
        ce_info.append([np.mean(losses), end_time - start_time])
        # final save
        logging.info("the final step is %s" % global_step)
        model_save_dir = os.path.join(args.output_dir, conf_dict["model_path"])
        model_path = os.path.join(model_save_dir, str(global_step))

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        fluid.dygraph.save_dygraph(net.state_dict(), model_path)
        logging.info("saving infer model in %s" % model_path)
        # used for continuous evaluation
        if args.enable_ce:
            card_num = get_cards()
            ce_loss = 0
            ce_time = 0
            try:
                ce_loss = ce_info[-1][0]
                ce_time = ce_info[-1][1]
            except:
                logging.info("ce info err!")
            print("kpis\teach_step_duration_%s_card%s\t%s" %
                  (args.task_name, card_num, ce_time))
            print("kpis\ttrain_loss_%s_card%s\t%f" %
                  (args.task_name, card_num, ce_loss))

        if args.do_test:
            # Get Feeder and Reader
            test_loader = fluid.io.DataLoader.from_generator(
                capacity=16,
                return_list=True,
                iterable=True,
                use_double_buffer=True)
            get_test_examples = simnet_process.get_reader("test")
            test_loader.set_sample_list_generator(
                paddle.batch(
                    get_test_examples, batch_size=args.batch_size),
                place)
            pred_list = []
            for left, pos_right in test_loader():
                left = fluid.layers.reshape(left, shape=[-1, 1])
                pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])
                net.eval()
                left = fluid.layers.reshape(left, shape=[-1, 1])
                pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])
                left_feat, pos_score = net(left, pos_right)
                pred = pos_score
                pred_list += list(pred.numpy())
            test_result = valid_and_test(pred_list, simnet_process, "test")
            if args.compute_accuracy:
                test_auc, test_acc = test_result
                logging.info("AUC of test is %f, Accuracy of test is %f" %
                             (test_auc, test_acc))
            else:
                test_auc = test_result
                logging.info("AUC of test is %f" % test_auc)


def test(conf_dict, args):
    """
    Evaluation Function
    """
    logging.info("start test process ...")
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):

        vocab = utils.load_vocab(args.vocab_path)
        simnet_process = reader.SimNetProcessor(args, vocab)
        test_loader = fluid.io.DataLoader.from_generator(
            capacity=16,
            return_list=True,
            iterable=True,
            use_double_buffer=True)
        get_test_examples = simnet_process.get_reader("test")
        test_loader.set_sample_list_generator(
            paddle.batch(
                get_test_examples, batch_size=args.batch_size), place)

        conf_dict['dict_size'] = len(vocab)
        conf_dict['seq_len'] = args.seq_len

        net = utils.import_class("./nets", conf_dict["net"]["module_name"],
                                 conf_dict["net"]["class_name"])(conf_dict)

        model, _ = load_dygraph(args.init_checkpoint)
        net.set_dict(model)
        metric = fluid.metrics.Auc(name="auc")
        pred_list = []
        with io.open(
                "predictions.txt", "w", encoding="utf8") as predictions_file:
            if args.task_mode == "pairwise":
                for left, pos_right in test_loader():
                    left = fluid.layers.reshape(left, shape=[-1, 1])
                    pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])

                    left_feat, pos_score = net(left, pos_right)
                    pred = pos_score

                    pred_list += list(
                        map(lambda item: float(item[0]), pred.numpy()))
                    predictions_file.write(u"\n".join(
                        map(lambda item: str((item[0] + 1) / 2), pred.numpy()))
                                           + "\n")

            else:
                for left, right in test_loader():
                    left = fluid.layers.reshape(left, shape=[-1, 1])
                    right = fluid.layers.reshape(right, shape=[-1, 1])
                    left_feat, pred = net(left, right)

                    pred_list += list(
                        map(lambda item: float(item[0]), pred.numpy()))
                    predictions_file.write(u"\n".join(
                        map(lambda item: str(np.argmax(item)), pred.numpy())) +
                                           "\n")

            if args.task_mode == "pairwise":
                pred_list = np.array(pred_list).reshape((-1, 1))
                pred_list = (pred_list + 1) / 2
                pred_list = np.hstack(
                    (np.ones_like(pred_list) - pred_list, pred_list))
            else:
                pred_list = np.array(pred_list)
            labels = simnet_process.get_test_label()

            metric.update(pred_list, labels)
            if args.compute_accuracy:
                acc = utils.get_accuracy(pred_list, labels, args.task_mode,
                                         args.lamda)
                logging.info("AUC of test is %f, Accuracy of test is %f" %
                             (metric.eval(), acc))
            else:
                logging.info("AUC of test is %f" % metric.eval())

        if args.verbose_result:
            utils.get_result_file(args)
            logging.info("test result saved in %s" %
                         os.path.join(os.getcwd(), args.test_result_path))


def infer(conf_dict, args):
    """
    run predict
    """
    logging.info("start test process ...")
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        vocab = utils.load_vocab(args.vocab_path)
        simnet_process = reader.SimNetProcessor(args, vocab)
        get_infer_examples = simnet_process.get_infer_reader
        infer_loader = fluid.io.DataLoader.from_generator(
            capacity=16,
            return_list=True,
            iterable=True,
            use_double_buffer=True)
        infer_loader.set_sample_list_generator(
            paddle.batch(
                get_infer_examples, batch_size=args.batch_size), place)

        conf_dict['dict_size'] = len(vocab)
        conf_dict['seq_len'] = args.seq_len

        net = utils.import_class("./nets", conf_dict["net"]["module_name"],
                                 conf_dict["net"]["class_name"])(conf_dict)
        model, _ = load_dygraph(args.init_checkpoint)
        net.set_dict(model)

        pred_list = []
        if args.task_mode == "pairwise":
            for left, pos_right in infer_loader():
                left = fluid.layers.reshape(left, shape=[-1, 1])
                pos_right = fluid.layers.reshape(pos_right, shape=[-1, 1])

                left_feat, pos_score = net(left, pos_right)
                pred = pos_score
                pred_list += list(
                    map(lambda item: str((item[0] + 1) / 2), pred.numpy()))

        else:
            for left, right in infer_loader():
                left = fluid.layers.reshape(left, shape=[-1, 1])
                pos_right = fluid.layers.reshape(right, shape=[-1, 1])
                left_feat, pred = net(left, right)
                pred_list += map(lambda item: str(np.argmax(item)),
                                 pred.numpy())

        with io.open(
                args.infer_result_path, "w", encoding="utf8") as infer_file:
            for _data, _pred in zip(simnet_process.get_infer_data(), pred_list):
                infer_file.write(_data + "\t" + _pred + "\n")
        logging.info("infer result saved in %s" %
                     os.path.join(os.getcwd(), args.infer_result_path))


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == "__main__":

    args = ArgConfig()
    args = args.build_conf()

    utils.print_arguments(args)
    check_cuda(args.use_cuda)
    check_version()
    utils.init_log("./log/TextSimilarityNet")
    conf_dict = config.SimNetConfig(args)
    if args.do_train:
        train(conf_dict, args)
    elif args.do_test:
        test(conf_dict, args)
    elif args.do_infer:
        infer(conf_dict, args)
    else:
        raise ValueError(
            "one of do_train and do_test and do_infer must be True")
