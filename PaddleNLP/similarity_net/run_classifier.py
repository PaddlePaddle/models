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

sys.path.append("..")

import paddle
import paddle.fluid as fluid
import numpy as np
import config
import utils
import reader
import models.matching.paddle_layers as layers

import logging

parser = argparse.ArgumentParser(__doc__)
model_g = utils.ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("config_path", str, None, "Path to the json file for EmoTect model config.")
model_g.add_arg("init_checkpoint", str, "examples/cnn_pointwise.json", "Init checkpoint to resume training from.")
model_g.add_arg("output_dir", str, None, "Directory path to save checkpoints")
model_g.add_arg("task_mode", str, None, "task mode: pairwise or pointwise")

train_g = utils.ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 10, "Number of epoches for training.")
train_g.add_arg("save_steps", int, 200, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 100, "The steps interval to evaluate model performance.")

log_g = utils.ArgumentGroup(parser, "logging", "logging related")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose_result", bool, True, "Whether to output verbose result.")
log_g.add_arg("test_result_path", str, "test_result", "Directory path to test result.")
log_g.add_arg("infer_result_path", str, "infer_result", "Directory path to infer result.")

data_g = utils.ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_data_dir", str, None, "Directory path to training data.")
data_g.add_arg("valid_data_dir", str, None, "Directory path to valid data.")
data_g.add_arg("test_data_dir", str, None, "Directory path to testing data.")
data_g.add_arg("infer_data_dir", str, None, "Directory path to infer data.")
data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training.")

run_type_g = utils.ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, False, "If set, use GPU for training.")
run_type_g.add_arg("task_name", str, None, "The name of task to perform sentiment classification.")
run_type_g.add_arg("do_train", bool, False, "Whether to perform training.")
run_type_g.add_arg("do_valid", bool, False, "Whether to perform dev.")
run_type_g.add_arg("do_test", bool, False, "Whether to perform testing.")
run_type_g.add_arg("do_infer", bool, False, "Whether to perform inference.")
run_type_g.add_arg("compute_accuracy", bool, False, "Whether to compute accuracy.")
run_type_g.add_arg("lamda", float, 0.91,
                   "When task_mode is pairwise, lamda is the threshold for calculating the accuracy.")

args = parser.parse_args()


def train(conf_dict, args):
    """
    train processic
    """
    # loading vocabulary
    vocab = utils.load_vocab(args.vocab_path)
    # get vocab size
    conf_dict['dict_size'] = len(vocab)
    # Get data layer
    data = layers.DataLayer()
    # Load network structure dynamically
    net = utils.import_class(
        "../models/matching", conf_dict["net"]["module_name"], conf_dict["net"]["class_name"])(conf_dict)
    # Load loss function dynamically
    loss = utils.import_class(
        "../models/matching/losses", conf_dict["loss"]["module_name"], conf_dict["loss"]["class_name"])(conf_dict)
    # Load Optimization method
    optimizer = utils.import_class(
        "../models/matching/optimizers", "paddle_optimizers", conf_dict["optimizer"]["class_name"])(conf_dict)
    # load auc method
    metric = fluid.metrics.Auc(name="auc")
    # Get device
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    simnet_process = reader.SimNetProcessor(args, vocab)
    if args.task_mode == "pairwise":
        # Build network
        left = data.ops(name="left", shape=[1], dtype="int64", lod_level=1)
        pos_right = data.ops(name="right", shape=[1], dtype="int64", lod_level=1)
        neg_right = data.ops(name="neg_right", shape=[1], dtype="int64", lod_level=1)
        left_feat, pos_score = net.predict(left, pos_right)

        # Get Feeder and Reader
        train_feeder = fluid.DataFeeder(place=place, feed_list=[left.name, pos_right.name, neg_right.name])
        train_reader = simnet_process.get_reader("train")
        if args.do_valid:
            valid_feeder = fluid.DataFeeder(place=place, feed_list=[left.name, pos_right.name])
            valid_reader = simnet_process.get_reader("valid")
            pred = pos_score
        # Save Infer model
        infer_program = fluid.default_main_program().clone(for_test=True)
        _, neg_score = net.predict(left, neg_right)
        avg_cost = loss.compute(pos_score, neg_score)
        avg_cost.persistable = True
    else:
        # Build network
        left = data.ops(name="left", shape=[1], dtype="int64", lod_level=1)
        right = data.ops(name="right", shape=[1], dtype="int64", lod_level=1)
        label = data.ops(name="label", shape=[1], dtype="int64", lod_level=0)
        left_feat, pred = net.predict(left, right)

        # Get Feeder and Reader
        train_feeder = fluid.DataFeeder(place=place, feed_list=[left.name, right.name, label.name])
        train_reader = simnet_process.get_reader("train")
        if args.do_valid:
            valid_feeder = fluid.DataFeeder(place=place, feed_list=[left.name, right.name])
            valid_reader = simnet_process.get_reader("valid")
        # Save Infer model
        infer_program = fluid.default_main_program().clone(for_test=True)
        avg_cost = loss.compute(pred, label)
        avg_cost.persistable = True

    # operate Optimization
    optimizer.ops(avg_cost)
    executor = fluid.Executor(place)
    executor.run(fluid.default_startup_program())
    # Get and run executor
    parallel_executor = fluid.ParallelExecutor(use_cuda=args.use_cuda, loss_name=avg_cost.name,
                                               main_program=fluid.default_main_program())
    # Get device number
    device_count = parallel_executor.device_count
    logging.info("device count: %d" % device_count)

    def valid_and_test(program, feeder, reader, process, mode="test"):
        """
        return auc and acc
        """
        # Get Batch Data
        batch_data = paddle.batch(reader, args.batch_size, drop_last=False)
        pred_list = []
        for data in batch_data():
            _pred = executor.run(program=program, feed=feeder.feed(data), fetch_list=[pred.name])
            pred_list += list(_pred)
        pred_list = np.vstack(pred_list)
        if mode == "test":
            label_list = process.get_test_label()
        elif mode == "valid":
            label_list = process.get_valid_label()
        if conf_dict['net']['class_name'] == 'MMDNN':
            pred_list = utils.deal_preds_of_mmdnn(conf_dict, pred_list)
        if args.task_mode == "pairwise":
            pred_list = np.hstack((np.ones_like(pred_list) - pred_list, pred_list))
        metric.reset()
        metric.update(pred_list, label_list)
        auc = metric.eval()
        if args.compute_accuracy:
            acc = utils.get_accuracy(pred_list, label_list, args.task_mode, args.lamda)
            return auc, acc
        else:
            return auc

    # run train
    logging.info("start train process ...")
    # set global step
    global_step = 0
    for epoch_id in range(args.epoch):
        losses = []
        # Get batch data iterator
        train_batch_data = paddle.batch(paddle.reader.shuffle(train_reader, buf_size=10000),
                                        args.batch_size, drop_last=False)
        start_time = time.time()
        for iter, data in enumerate(train_batch_data()):
            if len(data) < device_count:
                logging.info("the size of batch data is less than device_count(%d)" % device_count)
                continue
            global_step += 1
            avg_loss = parallel_executor.run([avg_cost.name], feed=train_feeder.feed(data))
            if args.do_valid and global_step % args.validation_steps == 0:

                valid_result = valid_and_test(program=infer_program, feeder=valid_feeder, reader=valid_reader,
                                              process=simnet_process, mode="valid")
                if args.compute_accuracy:
                    valid_auc, valid_acc = valid_result
                    logging.info("global_steps: %d, valid_auc: %f, valid_acc: %f" % (global_step, valid_auc, valid_acc))
                else:
                    valid_auc = valid_result
                    logging.info("global_steps: %d, valid_auc: %f" % (global_step, valid_auc))
            if global_step % args.save_steps == 0:
                model_save_dir = os.path.join(args.output_dir, conf_dict["model_path"])
                model_path = os.path.join(model_save_dir, str(global_step))

                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                if args.task_mode == "pairwise":
                    feed_var_names = [left.name, pos_right.name]
                    target_vars = [left_feat, pos_score]
                else:
                    feed_var_names = [left.name, right.name, ]
                    target_vars = [left_feat, pred]
                fluid.io.save_inference_model(
                    model_path, feed_var_names, target_vars, executor, infer_program)
                logging.info("saving infer model in %s" % model_path)
            losses.append(np.mean(avg_loss[0]))
        end_time = time.time()
        logging.info("epoch: %d, loss: %f, used time: %d sec" % (epoch_id, np.mean(losses), end_time - start_time))
    if args.do_test:
        if args.task_mode == "pairwise":
            # Get Feeder and Reader
            test_feeder = fluid.DataFeeder(place=place, feed_list=[left.name, pos_right.name])
            test_reader = simnet_process.get_reader("test")
        else:
            # Get Feeder and Reader
            test_feeder = fluid.DataFeeder(place=place, feed_list=[left.name, right.name])
            test_reader = simnet_process.get_reader("test")
        test_result = valid_and_test(program=infer_program, feeder=test_feeder, reader=test_reader,
                                     process=simnet_process, mode="test")
        if args.compute_accuracy:
            test_auc, test_acc = test_result
            logging.info("AUC of test is %f, Accuracy of test is %f" % (test_auc, test_acc))
        else:
            test_auc = test_result
            logging.info("AUC of test is %f" % test_auc)


def test(conf_dict, args):
    """
    run predict
    """
    vocab = utils.load_vocab(args.vocab_path)
    simnet_process = reader.SimNetProcessor(args, vocab)
    # load auc method
    metric = fluid.metrics.Auc(name="auc")
    with open("predictions.txt", "w") as predictions_file:
        # Get model path
        model_path = args.init_checkpoint
        # Get device
        if args.use_cuda:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        # Get executor
        executor = fluid.Executor(place=place)
        # Load model
        program, feed_var_names, fetch_targets = fluid.io.load_inference_model(model_path, executor)
        if args.task_mode == "pairwise":
            # Get Feeder and Reader
            feeder = fluid.DataFeeder(place=place, feed_list=feed_var_names, program=program)
            test_reader = simnet_process.get_reader("test")
        else:
            # Get Feeder and Reader
            feeder = fluid.DataFeeder(place=place, feed_list=feed_var_names, program=program)
            test_reader = simnet_process.get_reader("test")
        # Get batch data iterator
        batch_data = paddle.batch(test_reader, args.batch_size, drop_last=False)
        logging.info("start test process ...")
        pred_list = []
        for iter, data in enumerate(batch_data()):
            output = executor.run(program, feed=feeder.feed(data), fetch_list=fetch_targets)
            if args.task_mode == "pairwise":
                pred_list += list(map(lambda item: float(item[0]), output[1]))
                predictions_file.write("\n".join(map(lambda item: str(item[0]), output[1])) + "\n")
            else:
                pred_list += map(lambda item: item, output[1])
                predictions_file.write("\n".join(map(lambda item: str(np.argmax(item)), output[1])) + "\n")
        if conf_dict['net']['class_name'] == 'MMDNN':
            pred_list = utils.deal_preds_of_mmdnn(conf_dict, pred_list)
        if args.task_mode == "pairwise":
            pred_list = np.array(pred_list).reshape((-1, 1))
            pred_list = np.hstack((np.ones_like(pred_list) - pred_list, pred_list))
        else:
            pred_list = np.array(pred_list)
        labels = simnet_process.get_test_label()

        metric.update(pred_list, labels)
        if args.compute_accuracy:
            acc = utils.get_accuracy(pred_list, labels, args.task_mode, args.lamda)
            logging.info("AUC of test is %f, Accuracy of test is %f" % (metric.eval(), acc))
        else:
            logging.info("AUC of test is %f" % metric.eval())

    if args.verbose_result:
        utils.get_result_file(args)
        logging.info("test result saved in %s" % os.path.join(os.getcwd(), args.test_result_path))


def infer(args):
    """
    run predict
    """
    vocab = utils.load_vocab(args.vocab_path)
    simnet_process = reader.SimNetProcessor(args, vocab)
    # Get model path
    model_path = args.init_checkpoint
    # Get device
    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    # Get executor
    executor = fluid.Executor(place=place)
    # Load model
    program, feed_var_names, fetch_targets = fluid.io.load_inference_model(model_path, executor)
    if args.task_mode == "pairwise":
        # Get Feeder and Reader
        infer_feeder = fluid.DataFeeder(place=place, feed_list=feed_var_names, program=program)
        infer_reader = simnet_process.get_infer_reader
    else:
        # Get Feeder and Reader
        infer_feeder = fluid.DataFeeder(place=place, feed_list=feed_var_names, program=program)
        infer_reader = simnet_process.get_infer_reader
    # Get batch data iterator
    batch_data = paddle.batch(infer_reader, args.batch_size, drop_last=False)
    logging.info("start test process ...")
    preds_list = []
    for iter, data in enumerate(batch_data()):
        output = executor.run(program, feed=infer_feeder.feed(data), fetch_list=fetch_targets)
        if args.task_mode == "pairwise":
            preds_list += list(map(lambda item: str(item[0]), output[1]))
        else:
            preds_list += map(lambda item: str(np.argmax(item)), output[1])
    with open(args.infer_result_path, "w") as infer_file:
        for _data, _pred in zip(simnet_process.get_infer_data(), preds_list):
            infer_file.write(_data + "\t" + _pred + "\n")
    logging.info("infer result saved in %s" % os.path.join(os.getcwd(), args.infer_result_path))


def main(conf_dict, args):
    """
    main
    """
    if args.do_train:
        train(conf_dict, args)
    elif args.do_test:
        test(conf_dict, args)
    elif args.do_infer:
        infer(args)
    else:
        raise ValueError("one of do_train and do_test and do_infer must be True")


if __name__ == "__main__":
    utils.print_arguments(args)
    utils.init_log("./log/TextSimilarityNet")
    conf_dict = config.SimNetConfig(args)
    main(conf_dict, args)
