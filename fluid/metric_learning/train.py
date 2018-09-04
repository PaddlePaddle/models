import os
import sys
import math
import time
import argparse
import functools
import numpy as np
import paddle
import paddle.fluid as fluid
import models
from losses import tripletloss
from losses import quadrupletloss
from losses import emlloss
from losses.metrics import recall_topk
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('train_batch_size',   int,   80,                   "Minibatch size.")
add_arg('test_batch_size',    int,   10,                   "Minibatch size.")
add_arg('use_gpu',            bool,  True,                 "Whether to use GPU or not.")
add_arg('num_epochs',         int,   120,                  "Number of epochs.")
add_arg('image_shape',        str,   "3,224,224",          "Input image size.")
add_arg('model_save_dir',     str,   "output",             "Model save directory")
add_arg('with_mem_opt',       bool,  True,                 "Whether to use memory optimization or not.")
add_arg('pretrained_model',   str,   None,                 "Whether to use pretrained model.")
add_arg('checkpoint',         str,   None,                 "Whether to resume checkpoint.")
add_arg('lr',                 float, 0.1,                  "Set learning rate.")
add_arg('lr_strategy',        str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('model',              str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('loss_name',          str,   "tripletloss",        "Set the loss type to use.")
add_arg('samples_each_class', int,   2,                    "Samples each class.")
add_arg('num_gpus',           int,   8,                    "Number of gpus.")
add_arg('margin',             float, 0.1,                  "Parameter margin.")
add_arg('alpha',              float, 0.0,                  "Parameter alpha.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def optimizer_setting(params):
    ls = params["learning_strategy"]
    assert ls[
        "name"] == "piecewise_decay", "learning rate strategy must be {}, but got {}".format(
            "piecewise_decay", lr["name"])

    step = 10000
    bd = [step * e for e in ls["epochs"]]
    base_lr = params["lr"]
    lr = []
    lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))

    return optimizer


def train(args):
    # parameters from arguments
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    model_save_dir = args.model_save_dir
    loss_name = args.loss_name

    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()
    out = model.net(input=image, class_dim=200)

    if loss_name == "tripletloss":
        metricloss = tripletloss(
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            margin=0.1)
        cost_metric = metricloss.loss(out[0])
        avg_cost_metric = fluid.layers.mean(x=cost_metric)
    elif loss_name == "quadrupletloss":
        metricloss = quadrupletloss(
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            samples_each_class=args.samples_each_class,
            num_gpus=args.num_gpus,
            margin=args.margin)
        cost_metric = metricloss.loss(out[0])
        avg_cost_metric = fluid.layers.mean(x=cost_metric)
    elif loss_name == "emlloss":
        metricloss = emlloss(
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            samples_each_class=2,
            fea_dim=2048)
        cost_metric = metricloss.loss(out[0])
        avg_cost_metric = fluid.layers.mean(x=cost_metric)
    else:
        print("loss name is not supported!")
        exit()

    cost_cls = fluid.layers.cross_entropy(input=out[1], label=label)
    avg_cost_cls = fluid.layers.mean(x=cost_cls)
    acc_top1 = fluid.layers.accuracy(input=out[1], label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out[1], label=label, k=5)
    avg_cost = avg_cost_metric + args.alpha * avg_cost_cls

    test_program = fluid.default_main_program().clone(for_test=True)

    # parameters from model and arguments
    params = model.params
    params["lr"] = args.lr
    params["num_epochs"] = args.num_epochs
    params["learning_strategy"]["batch_size"] = args.train_batch_size
    params["learning_strategy"]["name"] = args.lr_strategy

    # initialize optimizer
    optimizer = optimizer_setting(params)
    opts = optimizer.minimize(avg_cost)

    global_lr = optimizer._global_learning_rate()

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = metricloss.train_reader
    test_reader = metricloss.test_reader
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    train_exe = fluid.ParallelExecutor(
        use_cuda=True if args.use_gpu else False, loss_name=avg_cost.name)

    fetch_list_train = [
        avg_cost_metric.name, avg_cost_cls.name, acc_top1.name, acc_top5.name,
        global_lr.name
    ]
    fetch_list_test = [out[0].name]

    if with_memory_optimization:
        fluid.memory_optimize(
            fluid.default_main_program(), skip_opt_set=set(fetch_list_train))

    for pass_id in range(params["num_epochs"]):
        train_info = [[], [], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss_metric, loss_cls, acc1, acc5, lr = train_exe.run(
                fetch_list_train, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss_metric = np.mean(np.array(loss_metric))
            loss_cls = np.mean(np.array(loss_cls))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            lr = np.mean(np.array(lr))
            train_info[0].append(loss_metric)
            train_info[1].append(loss_cls)
            train_info[2].append(acc1)
            train_info[3].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, lr {2}, loss_metric {3}, loss_cls {4}, acc1 {5}, acc5 {6}, time {7}".format(pass_id,  \
                      batch_id, lr, loss_metric, loss_cls, acc1, acc5, "%2.2f sec" % period))

        train_loss_metric = np.array(train_info[0]).mean()
        train_loss_cls = np.array(train_info[1]).mean()
        train_acc1 = np.array(train_info[2]).mean()
        train_acc5 = np.array(train_info[3]).mean()
        f = []
        l = []
        for batch_id, (data, label) in enumerate(test_reader()):
            t1 = time.time()
            [feas] = exe.run(test_program,
                             fetch_list=fetch_list_test,
                             feed=feeder.feed(data))
            f.append(feas)
            l.append(label)

            t2 = time.time()
            period = t2 - t1
            if batch_id % 10 == 0:
                recall = recall_topk(feas, label, k=1)
                print("Pass {0}, testbatch {1}, recall {2}, time {3}".format(pass_id,  \
                      batch_id, recall, "%2.2f sec" % period))

        f = np.vstack(f)
        l = np.vstack(l)
        recall = recall_topk(f, l, k=1)
        print("End pass {0}, train_loss_metric {1}, train_loss_cls {2}, train_acc1 {3}, train_acc5 {4}, test_recall {5}".format(pass_id,  \
              train_loss_metric, train_loss_cls, train_acc1, train_acc5, recall))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir + '/' + model_name,
                                  str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()
