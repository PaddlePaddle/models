from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import time
import logging
import argparse
import functools
import threading
import subprocess
import numpy as np
import paddle
import paddle.fluid as fluid
import models
import reader
from losses import TripletLoss
from losses import QuadrupletLoss
from losses import EmlLoss
from losses import NpairsLoss
from utility import add_arguments, print_arguments
from utility import fmt_time, recall_topk, get_gpu_num

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('embedding_size', int, 0, "Embedding size.")
add_arg('train_batch_size', int, 120, "Minibatch size.")
add_arg('test_batch_size', int, 50, "Minibatch size.")
add_arg('image_shape', str, "3,224,224", "input image size")
add_arg('class_dim', int, 11318, "Class number.")
add_arg('lr', float, 0.0001, "set learning rate.")
add_arg('lr_strategy', str, "piecewise_decay",	"Set the learning rate decay strategy.")
add_arg('lr_steps', str, "100000", "step of lr")
add_arg('total_iter_num', int, 100000, "total_iter_num")
add_arg('display_iter_step', int, 10, "display_iter_step.")
add_arg('test_iter_step', int, 5000, "test_iter_step.")
add_arg('save_iter_step', int, 5000, "save_iter_step.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('with_mem_opt', bool, True, "Whether to use memory optimization or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('checkpoint', str, None, "Whether to resume checkpoint.")
add_arg('model_save_dir', str, "output", "model save directory")
add_arg('loss_name', str, "triplet", "Set the loss type to use.")
add_arg('samples_each_class', int, 2, "samples_each_class.")
add_arg('margin', float, 0.1, "margin.")
add_arg('npairs_reg_lambda', float, 0.01, "npairs reg lambda.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]

def optimizer_setting(params):
    ls = params["learning_strategy"]
    assert ls["name"] == "piecewise_decay", \
           "learning rate strategy must be {}, \
           but got {}".format("piecewise_decay", lr["name"])

    bd = [int(e) for e in ls["lr_steps"].split(',')]
    base_lr = params["lr"]
    lr = [base_lr * (0.1 ** i) for i in range(len(bd) + 1)]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    return optimizer


def net_config(image, label, model, args, is_train):
    assert args.model in model_list, "{} is not in lists: {}".format(
        args.model, model_list)

    out = model.net(input=image, embedding_size=args.embedding_size)
    if not is_train:
        return None, out

    if args.loss_name == "triplet":
        metricloss = TripletLoss(
                margin=args.margin,
        )
    elif args.loss_name == "quadruplet":
        metricloss = QuadrupletLoss(
                train_batch_size = args.train_batch_size,
                samples_each_class = args.samples_each_class,
                margin=args.margin,
        )
    elif args.loss_name == "eml":
        metricloss = EmlLoss(
                train_batch_size = args.train_batch_size,
                samples_each_class = args.samples_each_class,
        )
    elif args.loss_name == "npairs":
        metricloss = NpairsLoss(
                train_batch_size = args.train_batch_size,
                samples_each_class = args.samples_each_class,
                reg_lambda = args.npairs_reg_lambda,
        )
    cost = metricloss.loss(out, label)
    avg_cost = fluid.layers.mean(x=cost)
    return avg_cost, out

def build_program(is_train, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model = models.__dict__[args.model]()
    with fluid.program_guard(main_prog, startup_prog):
        if is_train:
            queue_capacity = 64
            py_reader = fluid.layers.py_reader(
                capacity=queue_capacity,
                shapes=[[-1] + image_shape, [-1, 1]],
                lod_levels=[0, 0],
                dtypes=["float32", "int64"],
                use_double_buffer=True)
            image, label = fluid.layers.read_file(py_reader)
        else:
            image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        with fluid.unique_name.guard():
            avg_cost, out = net_config(image, label, model, args, is_train)
            if is_train:
                params = model.params
                params["lr"] = args.lr
                params["learning_strategy"]["lr_steps"] = args.lr_steps
                params["learning_strategy"]["name"] = args.lr_strategy
                optimizer = optimizer_setting(params)
                optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()
    """            
    if not is_train:
        main_prog = main_prog.clone(for_test=True)
    """
    if is_train:
        return py_reader, avg_cost, global_lr, out, label
    else: 
        return out, image, label


def train_async(args):
    # parameters from arguments

    logging.debug('enter train')
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    tmp_prog = fluid.Program()

    train_py_reader, train_cost, global_lr, train_feas, train_label = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    test_feas, image, label = build_program(
        is_train=False,
        main_prog=tmp_prog,
        startup_prog=startup_prog,
        args=args)
    test_prog = tmp_prog.clone(for_test=True)

    train_fetch_list = [global_lr.name, train_cost.name, train_feas.name, train_label.name]
    test_fetch_list = [test_feas.name]

    if args.with_mem_opt:
        fluid.memory_optimize(train_prog, skip_opt_set=set(train_fetch_list))

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(startup_prog)

    logging.debug('after run startup program')

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint, main_program=train_prog)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    devicenum = get_gpu_num()
    assert (args.train_batch_size % devicenum) == 0
    train_batch_size = args.train_batch_size / devicenum
    test_batch_size = args.test_batch_size
    
    train_reader = paddle.batch(reader.train(args), batch_size=train_batch_size, drop_last=True)
    test_reader = paddle.batch(reader.test(args), batch_size=test_batch_size, drop_last=False)
    test_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_py_reader.decorate_paddle_reader(train_reader)

    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=args.use_gpu,
        loss_name=train_cost.name)

    totalruntime = 0
    train_py_reader.start()
    iter_no = 0
    train_info = [0, 0, 0]
    while iter_no <= args.total_iter_num:
        t1 = time.time()
        lr, loss, feas, label = train_exe.run(fetch_list=train_fetch_list)
        t2 = time.time()
        period = t2 - t1
        lr = np.mean(np.array(lr))
        train_info[0] += np.mean(np.array(loss))
        train_info[1] += recall_topk(feas, label, k=1)
        train_info[2] += 1
        if iter_no % args.display_iter_step == 0:
            avgruntime = totalruntime / args.display_iter_step
            avg_loss = train_info[0] / train_info[2]
            avg_recall = train_info[1] / train_info[2]
            print("[%s] trainbatch %d, lr %.6f, loss %.6f, "\
                    "recall %.4f, time %2.2f sec" % \
                    (fmt_time(), iter_no, lr, avg_loss, avg_recall, avgruntime))
            sys.stdout.flush()
            totalruntime = 0
        if iter_no % 1000 == 0:
            train_info = [0, 0, 0]

        totalruntime += period
        
        if iter_no % args.test_iter_step == 0 and iter_no != 0:
            f, l = [], []
            for batch_id, data in enumerate(test_reader()):
                t1 = time.time()
                [feas] = exe.run(test_prog, fetch_list = test_fetch_list, feed=test_feeder.feed(data))
                label = np.asarray([x[1] for x in data])
                f.append(feas)
                l.append(label)

                t2 = time.time()
                period = t2 - t1
                if batch_id % 20 == 0:
                    print("[%s] testbatch %d, time %2.2f sec" % \
                            (fmt_time(), batch_id, period))

            f = np.vstack(f)
            l = np.hstack(l)
            recall = recall_topk(f, l, k=1)
            print("[%s] test_img_num %d, trainbatch %d, test_recall %.5f" % \
                    (fmt_time(), len(f), iter_no, recall))
            sys.stdout.flush()

        if iter_no % args.save_iter_step == 0 and iter_no != 0:
            model_path = os.path.join(model_save_dir + '/' + model_name,
                                      str(iter_no))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            fluid.io.save_persistables(exe, model_path, main_program=train_prog)

        iter_no += 1

def initlogging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    loglevel = logging.DEBUG
    logging.basicConfig(
        level=loglevel,
        # logger.BASIC_FORMAT,
        format=
        "%(levelname)s:%(filename)s[%(lineno)s] %(name)s:%(funcName)s->%(message)s",
        datefmt='%a, %d %b %Y %H:%M:%S')

def main():
    args = parser.parse_args()
    print_arguments(args)
    train_async(args)


if __name__ == '__main__':
    main()
