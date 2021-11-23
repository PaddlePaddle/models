# coding=utf-8

import os
import numpy as np
import time
import sys
import paddle as paddle
import paddle.fluid as fluid
import reader
import argparse
import functools
# import paddle.fluid.layers.ops as ops
from utility import add_arguments, print_arguments
from squeezenet import squeeze_net
# from paddle.v2.fluid.initializer import init_on_cpu
from paddle.fluid.layers import learning_rate_scheduler
import pdb
import psutil
from meliae import scanner

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size', int, 128, "Minibatch size.")
add_arg('with_mem_opt', bool, True,
        "Whether to use memory optimization or not.")
add_arg('parallel_exe', bool, False,
        "Whether to use ParallelExecutor to train or not.")
add_arg('init_model', str, None, "Whether to use initialized model.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('lr_strategy', str, 'poly_decay',
        "Set the learning rate decay strategy.")
add_arg('model', str, "", "Set the network to use.")
add_arg('train_list', str, "data/ILSVRC2012/train_list.txt",
        "name list of images used in training process.")
add_arg('test_list', str, "data/ILSVRC2012/val_list.txt",
        "name list of images used in testing process.")


def resolve_caffe_model(pretrain_model):
    '''Resolve the pretrained model parameters for finetuning.
    
    Args:
        pretrain_model: The pretrained model path.
    
    Returns:
        weights_dict: The resolved model parameters as dictionary format.
    '''
    items = os.listdir(pretrain_model)

    weights_dict = {}
    for item in items:
        param_name = item.split('.')[0]
        param_value = np.load(os.path.join(pretrain_model, item))
        weights_dict[param_name] = param_value
    return weights_dict


def train_parallel_do(
        args,
        learning_rate,
        batch_size,
        num_passes,
        init_model=None,
        pretrained_model=None,
        model_save_dir='models',
        parallel=True,
        use_nccl=True,
        lr_strategy=None, ):
    class_dim = 1000
    image_shape = [3, 227, 227]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=use_nccl)

        with pd.do():
            image_ = pd.read_input(image)
            label_ = pd.read_input(label)
            out = squeeze_net(img=image_, class_dim=class_dim)

            cost = fluid.layers.cross_entropy(input=out, label=label_)
            avg_cost = fluid.layers.mean(x=cost)
            acc_top1 = fluid.layers.accuracy(input=out, label=label_, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label_, k=5)
            pd.write_output(avg_cost)
            pd.write_output(acc_top1)
            pd.write_output(acc_top5)

        avg_cost, acc_top1, acc_top5 = pd()
        avg_cost = fluid.layers.mean(x=avg_cost)
        acc_top1 = fluid.layers.mean(x=acc_top1)
        acc_top5 = fluid.layers.mean(x=acc_top5)
    else:
        out = squeeze_net(img=image, class_dim=class_dim)

        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    inference_program = fluid.default_main_program().clone(for_test=True)
    if lr_strategy is None:
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate,
            regularization=fluid.regularizer.L2Decay(2e-4))

    elif "piecewise_decay" in lr_strategy:
        bd = lr_strategy["piecewise_decay"]["bd"]
        lr = lr_strategy["piecewise_decay"]["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        total_step = lr_strategy["poly_decay"]["total_step"]
        power = lr_strategy["poly_decay"]["power"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate_scheduler.polynomial_decay(
                learning_rate, total_step, end_learning_rate=1e-12,
                power=power),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(2e-4))

    opts = optimizer.minimize(avg_cost)
    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        weights_dict = resolve_caffe_model(args.init_model)
        for k, v in weights_dict.items():
            _tensor = fluid.global_scope().find_var(k).get_tensor()
            _shape = np.array(_tensor).shape
            _tensor.set(v, place)
    if pretrained_model is not None:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = paddle.batch(reader=reader.train(), batch_size=batch_size)
    test_reader = paddle.batch(reader=reader.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    for pass_id in range(num_passes):
        info = psutil.virtual_memory()
        print "rss: {}".format(psutil.Process(os.getpid()).memory_info().rss)
        print "total mem: {}".format(info.total)
        print "mem%: {}".format(info.percent)
        scanner.dump_all_objects('output/logs/dump_{}.txt'.format(pass_id))
        train_info = [[], [], []]
        test_info = [[], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss, acc1, acc5 = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost, acc_top1, acc_top5])
            t2 = time.time()
            period = t2 - t1
            train_info[0].append(loss[0])
            train_info[1].append(acc1[0])
            train_info[2].append(acc5[0])
            if batch_id % 10 == 0:
                print("Train: Pass: {}, Batch: {}, Loss: {:.10f},"
                      " Acc1: {:.4f}, Acc5: {:.4f}, Time: {}".format(
                          pass_id, batch_id, loss[0], acc1[0], acc5[0],
                          "{:.2f} sec".format(period)))
                sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        for data in test_reader():
            t1 = time.time()
            loss, acc1, acc5 = exe.run(
                inference_program,
                feed=feeder.feed(data),
                fetch_list=[avg_cost, acc_top1, acc_top5])
            t2 = time.time()
            period = t2 - t1
            test_info[0].append(loss[0])
            test_info[1].append(acc1[0])
            test_info[2].append(acc5[0])
            if batch_id % 10 == 0:
                print("TEST: Pass: {}, Batch: {}, Loss: {:.10f},"
                      "Acc1: {:.4f}, Acc5: {:.4f}, Time: {}".format(
                          pass_id, batch_id, loss[0], acc1[0], acc5[0],
                          "{.2f} sec".format(period)))
                sys.stdout.flush()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print(
            "End pass: {}, Train_Loss: {:.6f}, Train_Acc1: {:.4f}, Train_Acc5 {:.4f},"
            "Test_Loss: {:.6f}, Test_Acc1: {:.4f}, Test_Acc5: {:.4f}".format(
                pass_id, train_loss, train_acc1, train_acc5, test_loss,
                test_acc1, test_acc5))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir + '/' + args.model,
                                  str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)


def train_parallel_exe(
        args,
        learning_rate,
        batch_size,
        num_passes,
        init_model=None,
        pretrained_model=None,
        model_save_dir='models',
        parallel=True,
        use_nccl=True,
        lr_strategy=None, ):
    class_dim = 1000
    image_shape = [3, 227, 227]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    out = squeeze_net(img=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    avg_cost = fluid.layers.mean(x=cost)

    test_program = fluid.default_main_program().clone(for_test=True)

    if lr_strategy is None:
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate,
            regularization=fluid.regularizer.L2Decay(2e-4))

    elif "piecewise_decay" in lr_strategy:
        bd = lr_strategy["piecewise_decay"]["bd"]
        lr = lr_strategy["piecewise_decay"]["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        total_step = lr_strategy["poly_decay"]["total_step"]
        power = lr_strategy["poly_decay"]["power"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate_scheduler.polynomial_decay(
                learning_rate, total_step, end_learning_rate=1e-12,
                power=power),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(2e-4))

    opts = optimizer.minimize(avg_cost)

    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        weights_dict = resolve_caffe_model(args.init_model)
        for k, v in weights_dict.items():
            _tensor = fluid.global_scope().find_var(k).get_tensor()
            _shape = np.array(_tensor).shape
            _tensor.set(v, place)
    if pretrained_model is not None:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_reader = paddle.batch(reader.train(), batch_size=batch_size)
    test_reader = paddle.batch(reader.test(), batch_size=batch_size)

    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=avg_cost.name)
    test_exe = fluid.ParallelExecutor(
        use_cuda=True, main_program=test_program, share_vars_from=train_exe)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    for pass_id in range(num_passes):
        train_info = [[], [], []]
        test_info = [[], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss, acc1, acc5 = train_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, loss {2},"
                      "acc1 {3}, acc5 {4} time {5}".format(
                          pass_id, batch_id, loss, acc1, acc5,
                          "{.2f} sec".format(period)))
                sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        for data in test_reader():
            t1 = time.time()
            loss, acc1, acc5 = test_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            test_info[0].append(loss)
            test_info[1].append(acc1)
            test_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0},testbatch {1},loss {2}, "
                      "acc1 {3},acc5 {4},time {5}".format(
                          pass_id, batch_id, loss, acc1, acc5,
                          "{.2f} sec".format(period)))
                sys.stdout.flush()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3},"
              "test_loss {4}, test_acc1 {5}, test_acc5 {6}".format(
                  pass_id, train_loss, train_acc1, train_acc5, test_loss,
                  test_acc1, test_acc5))
        sys.stdout.flush()
        model_path = os.path.join(model_save_dir + '/' + args.model,
                                  str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    total_images = 1281167
    batch_size = args.batch_size
    step = int(total_images / batch_size + 1)
    num_epochs = 170
    total_step = step * num_epochs

    learning_rate_mode = args.lr_strategy
    lr_strategy = {}
    if learning_rate_mode == "piecewise_decay":
        epoch_points = [30, 60, 90]
        bd = [e * step for e in epoch_points]
        lr = [0.1, 0.01, 0.001, 0.0001]
        lr_strategy[learning_rate_mode] = {"bd": bd, "lr": lr}
    elif learning_rate_mode == "poly_decay":
        lr_strategy[learning_rate_mode] = {
            "total_step": total_step,
            "power": 1.0
        }
    else:
        lr_strategy = None

    use_nccl = False
    method = train_parallel_exe if args.parallel_exe else train_parallel_do
    init_model = args.init_model if args.init_model else None
    pretrained_model = args.pretrained_model if args.pretrained_model else None
    method(
        args,
        learning_rate=0.04,
        batch_size=batch_size,
        num_passes=num_epochs,
        init_model=init_model,
        pretrained_model=pretrained_model,
        parallel=True,
        use_nccl=use_nccl,
        lr_strategy=lr_strategy)
