import os
import numpy as np
import time
import sys
import paddle.v2 as paddle
import paddle.fluid as fluid
from se_resnext import SE_ResNeXt
import reader

import argparse
import functools
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',   int,  256, "Minibatch size.")
add_arg('num_layers',   int,  50,  "How many layers for SE-ResNeXt model.")
add_arg('with_mem_opt', bool, True, "Whether to use memory optimization or not.")
add_arg('parallel_exe', bool, True, "Whether to use ParallelExecutor to train or not.")

def train_paralle_do(args,
                     learning_rate,
                     batch_size,
                     num_passes,
                     init_model=None,
                     model_save_dir='model',
                     parallel=True,
                     use_nccl=True,
                     lr_strategy=None,
                     layers=50):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=use_nccl)

        with pd.do():
            image_ = pd.read_input(image)
            label_ = pd.read_input(label)
            out = SE_ResNeXt(input=image_, class_dim=class_dim, layers=layers)
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
        out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    if lr_strategy is None:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        bd = lr_strategy["bd"]
        lr = lr_strategy["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    inference_program = fluid.default_main_program().clone(for_test=True)

    opts = optimizer.minimize(avg_cost)
    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())
        fluid.memory_optimize(inference_program)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        fluid.io.load_persistables(exe, init_model)

    train_reader = paddle.batch(reader.train(), batch_size=batch_size)
    test_reader = paddle.batch(reader.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    for pass_id in range(num_passes):
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
                print("Pass {0}, trainbatch {1}, loss {2}, \
                       acc1 {3}, acc5 {4} time {5}"
                                                   .format(pass_id, \
                       batch_id, loss[0], acc1[0], acc5[0], \
                       "%2.2f sec" % period))
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
                print("Pass {0},testbatch {1},loss {2}, \
                       acc1 {3},acc5 {4},time {5}"
                                                  .format(pass_id, \
                       batch_id, loss[0], acc1[0], acc5[0], \
                       "%2.2f sec" % period))
                sys.stdout.flush()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, \
               test_loss {4}, test_acc1 {5}, test_acc5 {6}"
                                                           .format(pass_id, \
              train_loss, train_acc1, train_acc5, test_loss, test_acc1, \
              test_acc5))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir, str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)

def train_parallel_exe(args,
                       learning_rate,
                       batch_size,
                       num_passes,
                       init_model=None,
                       model_save_dir='model',
                       parallel=True,
                       use_nccl=True,
                       lr_strategy=None,
                       layers=50):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    avg_cost = fluid.layers.mean(x=cost)

    test_program = fluid.default_main_program().clone(for_test=True)

    if lr_strategy is None:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        bd = lr_strategy["bd"]
        lr = lr_strategy["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    opts = optimizer.minimize(avg_cost)

    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())
        fluid.memory_optimize(test_program)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        fluid.io.load_persistables(exe, init_model)

    train_reader = paddle.batch(reader.train(), batch_size=batch_size)
    test_reader = paddle.batch(reader.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=avg_cost.name)
    test_exe = fluid.ParallelExecutor(
        use_cuda=True,
        main_program=test_program,
        share_vars_from=train_exe)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    for pass_id in range(num_passes):
        train_info = [[], [], []]
        test_info = [[], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss, acc1, acc5 = train_exe.run(
                fetch_list,
                feed_dict=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, \
                       acc1 {3}, acc5 {4} time {5}"
                                                   .format(pass_id, \
                       batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period))
                sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        for data in test_reader():
            t1 = time.time()
            loss, acc1, acc5 = test_exe.run(
                fetch_list,
                feed_dict=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            test_info[0].append(loss)
            test_info[1].append(acc1)
            test_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0},testbatch {1},loss {2}, \
                       acc1 {3},acc5 {4},time {5}"
                                                  .format(pass_id, \
                       batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period))
                sys.stdout.flush()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, \
               test_loss {4}, test_acc1 {5}, test_acc5 {6}"
                                                           .format(pass_id, \
              train_loss, train_acc1, train_acc5, test_loss, test_acc1, \
              test_acc5))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir, str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)




if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    epoch_points = [30, 60, 90]
    total_images = 1281167
    batch_size = args.batch_size
    step = int(total_images / batch_size + 1)
    bd = [e * step for e in epoch_points]
    lr = [0.1, 0.01, 0.001, 0.0001]

    lr_strategy = {"bd": bd, "lr": lr}

    use_nccl = True
    # layers: 50, 152
    layers = args.num_layers
    method = train_parallel_exe if args.parallel_exe else train_parallel_do
    method(args,
           learning_rate=0.1,
           batch_size=batch_size,
           num_passes=120,
           init_model=None,
           parallel=True,
           use_nccl=True,
           lr_strategy=lr_strategy,
           layers=layers)
