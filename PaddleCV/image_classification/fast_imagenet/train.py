# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import cProfile
import time
import os
import traceback

import numpy as np
import math
import reader
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.transpiler.distribute_transpiler as distribute_transpiler

import sys
sys.path.append("..")
from utility import add_arguments, print_arguments
import functools
from models.fast_imagenet import FastImageNet, lr_decay
import utils


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('total_images',     int,   1281167,              "Training image number.")
    add_arg('num_epochs',       int,   120,                  "number of epochs.")
    add_arg('image_shape',      str,   "3,224,224",          "input image size")
    add_arg('model_save_dir',   str,   "output",             "model save directory")
    add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
    add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
    add_arg('lr',               float, 1.0,                  "set learning rate.")
    add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
    add_arg('data_dir',         str,   "./data/ILSVRC2012",  "The ImageNet dataset root dir.")
    add_arg('model_category',   str,   "models",             "Whether to use models_name or not, valid value:'models','models_name'" )
    add_arg('fp16',             bool,  False,                "Enable half precision training with fp16." )
    add_arg('scale_loss',       float, 1.0,                  "Scale loss for fp16." )
    # for distributed
    add_arg('start_test_pass',  int,    0,                  "Start test after x passes.")
    add_arg('num_threads',      int,    8,                  "Use num_threads to run the fluid program.")
    add_arg('reduce_strategy',  str,    "allreduce",        "Choose from reduce or allreduce.")
    add_arg('log_period',       int,    30,                 "Print period, defualt is 5.")
    add_arg('best_acc5',        float,  0.93,               "The best acc5, default is 93%.")
    # yapf: enable
    args = parser.parse_args()
    return args


DEVICE_NUM = fluid.core.get_cuda_device_count()


def test_parallel(exe, test_args, args, test_reader, feeder, bs):
    acc_evaluators = []
    for i in xrange(len(test_args[2])):
        acc_evaluators.append(fluid.metrics.Accuracy())

    to_fetch = [v.name for v in test_args[2]]
    batch_id = 0
    start_ts = time.time()
    for batch_id, data in enumerate(test_reader()):
        acc_rets = exe.run(fetch_list=to_fetch, feed=feeder.feed(data))
        ret_result = [np.mean(np.array(ret)) for ret in acc_rets]
        print("Test batch: [%d], acc_rets: [%s]" % (batch_id, ret_result))
        for i, e in enumerate(acc_evaluators):
            e.update(value=np.array(acc_rets[i]), weight=bs)
    num_samples = batch_id * bs * DEVICE_NUM
    print_train_time(start_ts, time.time(), num_samples)

    return [e.eval() for e in acc_evaluators]


def build_program(args,
                  is_train,
                  main_prog,
                  startup_prog,
                  py_reader_startup_prog,
                  sz,
                  trn_dir,
                  bs,
                  min_scale,
                  rect_val=False):

    dshape = [3, sz, sz]
    class_dim = 1000
    pyreader = None
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            if is_train:
                with fluid.program_guard(main_prog, py_reader_startup_prog):
                    with fluid.unique_name.guard():
                        pyreader = fluid.layers.py_reader(
                            capacity=bs * DEVICE_NUM,
                            shapes=([-1] + dshape, (-1, 1)),
                            dtypes=('uint8', 'int64'),
                            name="train_reader_" + str(sz)
                            if is_train else "test_reader_" + str(sz),
                            use_double_buffer=True)
                input, label = fluid.layers.read_file(pyreader)
            else:
                input = fluid.data(
                    name="image", shape=[None, 3, 244, 244], dtype="uint8")
                label = fluid.data(name="label", shape=[None, 1], dtype="int64")
            cast_img_type = "float16" if args.fp16 else "float32"
            cast = fluid.layers.cast(input, cast_img_type)
            img_mean = fluid.layers.create_global_var(
                [3, 1, 1],
                0.0,
                cast_img_type,
                name="img_mean",
                persistable=True)
            img_std = fluid.layers.create_global_var(
                [3, 1, 1], 0.0, cast_img_type, name="img_std", persistable=True)
            # image = (image - (mean * 255.0)) / (std * 255.0)
            t1 = fluid.layers.elementwise_sub(cast, img_mean, axis=1)
            t2 = fluid.layers.elementwise_div(t1, img_std, axis=1)

            model = FastImageNet(is_train=is_train)
            predict = model.net(t2, class_dim=class_dim, img_size=sz)
            cost, pred = fluid.layers.softmax_with_cross_entropy(
                predict, label, return_softmax=True)
            if args.scale_loss > 1:
                avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
            else:
                avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=pred, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=pred, label=label, k=5)

            # configure optimize
            optimizer = None
            if is_train:
                total_images = args.total_images
                lr = args.lr

                epochs = [(0, 7), (7, 13), (13, 22), (22, 25), (25, 28)]
                bs_epoch = [bs * DEVICE_NUM for bs in [224, 224, 96, 96, 50]]
                bs_scale = [bs * 1.0 / bs_epoch[0] for bs in bs_epoch]
                lrs = [(lr, lr * 2), (lr * 2, lr / 4),
                       (lr * bs_scale[2], lr / 10 * bs_scale[2]),
                       (lr / 10 * bs_scale[2], lr / 100 * bs_scale[2]),
                       (lr / 100 * bs_scale[4], lr / 1000 * bs_scale[4]),
                       lr / 1000 * bs_scale[4]]

                boundaries, values = lr_decay(lrs, epochs, bs_epoch,
                                              total_images)

                optimizer = fluid.optimizer.Momentum(
                    learning_rate=fluid.layers.piecewise_decay(
                        boundaries=boundaries, values=values),
                    momentum=0.9)
                if args.fp16:
                    params_grads = optimizer.backward(avg_cost)
                    master_params_grads = utils.create_master_params_grads(
                        params_grads, main_prog, startup_prog, args.scale_loss)
                    optimizer.apply_gradients(master_params_grads)
                    utils.master_param_to_train_param(master_params_grads,
                                                      params_grads, main_prog)
                else:
                    optimizer.minimize(avg_cost)

    return avg_cost, optimizer, [batch_acc1, batch_acc5], pyreader


def refresh_program(args,
                    epoch,
                    sz,
                    trn_dir,
                    bs,
                    val_bs,
                    need_update_start_prog=False,
                    min_scale=0.08,
                    rect_val=False):
    print(
        'refresh program: epoch: [%d], image size: [%d], trn_dir: [%s], batch_size:[%d]'
        % (epoch, sz, trn_dir, bs))
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    py_reader_startup_prog = fluid.Program()

    train_args = build_program(args, True, train_prog, startup_prog,
                               py_reader_startup_prog, sz, trn_dir, bs,
                               min_scale)
    test_args = build_program(
        args,
        False,
        test_prog,
        startup_prog,
        py_reader_startup_prog,
        sz,
        trn_dir,
        val_bs,
        min_scale,
        rect_val=rect_val)

    place = fluid.CUDAPlace(0)
    startup_exe = fluid.Executor(place)
    startup_exe.run(py_reader_startup_prog)

    if need_update_start_prog:
        startup_exe.run(startup_prog)
        conv2d_w_vars = [
            var for var in startup_prog.global_block().vars.values()
            if var.name.startswith('conv2d_')
        ]
        for var in conv2d_w_vars:
            #torch_w = torch.empty(var.shape)
            #kaiming_np = torch.nn.init.kaiming_normal_(torch_w, mode='fan_out', nonlinearity='relu').numpy()
            shape = var.shape
            if not shape or len(shape) == 0:
                fan_out = 1
            elif len(shape) == 1:
                fan_out = shape[0]
            elif len(shape) == 2:
                fan_out = shape[1]
            else:
                fan_out = shape[0] * np.prod(shape[2:])
            std = math.sqrt(2.0 / fan_out)
            kaiming_np = np.random.normal(0, std, var.shape)
            tensor = fluid.global_scope().find_var(var.name).get_tensor()
            if args.fp16:
                tensor.set(np.array(
                    kaiming_np, dtype="float16").view(np.uint16),
                           place)
            else:
                tensor.set(np.array(kaiming_np, dtype="float32"), place)

        np_tensors = {}
        np_tensors["img_mean"] = np.array(
            [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0]).astype(
                "float16" if args.fp16 else "float32").reshape((3, 1, 1))
        np_tensors["img_std"] = np.array(
            [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]).astype(
                "float16" if args.fp16 else "float32").reshape((3, 1, 1))
        for vname, np_tensor in np_tensors.items():
            var = fluid.global_scope().find_var(vname)
            if args.fp16:
                var.get_tensor().set(np_tensor.view(np.uint16), place)
            else:
                var.get_tensor().set(np_tensor, place)

    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = args.num_threads
    strategy.allow_op_delay = False
    strategy.num_iteration_per_drop_scope = 1
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy(
    ).ReduceStrategy.AllReduce

    avg_loss = train_args[0]
    train_exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        main_program=train_prog,
        exec_strategy=strategy,
        build_strategy=build_strategy)
    test_exe = fluid.ParallelExecutor(
        True, main_program=test_prog, share_vars_from=train_exe)

    return train_args, test_args, test_prog, train_exe, test_exe


def prepare_reader(epoch_id, train_py_reader, train_bs, val_bs, trn_dir,
                   img_dim, min_scale, rect_val, args):
    train_reader = reader.train(
        traindir="%s/%strain" % (args.data_dir, trn_dir),
        sz=img_dim,
        min_scale=min_scale,
        shuffle_seed=epoch_id + 1)
    train_py_reader.decorate_paddle_reader(
        fluid.io.batch(
            train_reader, batch_size=train_bs))

    test_reader = reader.test(
        valdir="%s/%svalidation" % (args.data_dir, trn_dir),
        bs=val_bs * DEVICE_NUM,
        sz=img_dim,
        rect_val=rect_val)
    test_batched_reader = fluid.io.batch(
        test_reader, batch_size=val_bs * DEVICE_NUM)

    return test_batched_reader


def train_parallel(args):
    over_all_start = time.time()
    test_prog = fluid.Program()

    exe = None
    test_exe = None
    train_args = None
    test_args = None
    ## dynamic batch size, image size...
    bs = 224
    val_bs = 64
    trn_dir = "160/"
    img_dim = 128
    min_scale = 0.08
    rect_val = False
    for epoch_id in range(args.num_epochs):
        # refresh program
        if epoch_id == 0:
            train_args, test_args, test_prog, exe, test_exe = refresh_program(
                args,
                epoch_id,
                sz=img_dim,
                trn_dir=trn_dir,
                bs=bs,
                val_bs=val_bs,
                need_update_start_prog=True)
        elif epoch_id == 13:  #13
            bs = 96
            trn_dir = "352/"
            img_dim = 224
            min_scale = 0.087
            train_args, test_args, test_prog, exe, test_exe = refresh_program(
                args,
                epoch_id,
                sz=img_dim,
                trn_dir=trn_dir,
                bs=bs,
                val_bs=val_bs,
                min_scale=min_scale)
        elif epoch_id == 25:  #25
            bs = 50
            val_bs = 8
            trn_dir = ""
            img_dim = 288
            min_scale = 0.5
            rect_val = True
            train_args, test_args, test_prog, exe, test_exe = refresh_program(
                args,
                epoch_id,
                sz=img_dim,
                trn_dir=trn_dir,
                bs=bs,
                val_bs=val_bs,
                min_scale=min_scale,
                rect_val=rect_val)
        else:
            pass

        avg_loss = train_args[0]
        num_samples = 0
        iters = 0
        start_time = time.time()
        train_py_reader = train_args[3]
        test_reader = prepare_reader(
            epoch_id,
            train_py_reader,
            bs,
            val_bs,
            trn_dir,
            img_dim=img_dim,
            min_scale=min_scale,
            rect_val=rect_val,
            args=args)
        train_py_reader.start()  # start pyreader
        batch_start_time = time.time()
        while True:
            fetch_list = [avg_loss.name]
            acc_name_list = [v.name for v in train_args[2]]
            fetch_list.extend(acc_name_list)
            fetch_list.append("learning_rate")
            if iters % args.log_period == 0:
                should_print = True
            else:
                should_print = False

            fetch_ret = []
            try:
                if should_print:
                    fetch_ret = exe.run(fetch_list)
                else:
                    exe.run([])
            except fluid.core.EOFException as eof:
                print("Finish current epoch, will reset pyreader...")
                train_py_reader.reset()
                break
            except fluid.core.EnforceNotMet as ex:
                traceback.print_exc()
                exit(1)

            num_samples += bs * DEVICE_NUM

            if should_print:
                fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                print(
                    "Epoch %d, batch %d, loss %s, accucacys: %s, learning_rate %s, py_reader queue_size: %d, avg batch time: %0.4f secs"
                    % (epoch_id, iters, fetched_data[0], fetched_data[1:-1],
                       fetched_data[-1], train_py_reader.queue.size(),
                       (time.time() - batch_start_time) * 1.0 /
                       args.log_period))
                batch_start_time = time.time()
            iters += 1

        print_train_time(start_time, time.time(), num_samples)
        feed_list = [
            test_prog.global_block().var(varname)
            for varname in ("image", "label")
        ]
        test_feeder = fluid.DataFeeder(
            feed_list=feed_list, place=fluid.CUDAPlace(0))
        test_ret = test_parallel(test_exe, test_args, args, test_reader,
                                 test_feeder, val_bs)
        test_acc1, test_acc5 = [np.mean(np.array(v)) for v in test_ret]
        print("Epoch: %d, Test Accuracy: %s, Spend %.2f hours\n" %
              (epoch_id, [test_acc1, test_acc5],
               (time.time() - over_all_start) / 3600))
        if np.mean(np.array(test_ret[1])) > args.best_acc5:
            print("Achieve the best top-1 acc %f, top-5 acc: %f" %
                  (test_acc1, test_acc5))
            break

    print("total train time: ", time.time() - over_all_start)


def print_train_time(start_time, end_time, num_samples):
    train_elapsed = end_time - start_time
    examples_per_sec = num_samples / train_elapsed
    print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
          (num_samples, train_elapsed, examples_per_sec))


def print_paddle_envs():
    print('----------- Configuration envs -----------')
    print("DEVICE_NUM: %d" % DEVICE_NUM)
    for k in os.environ:
        if "PADDLE_" in k:
            print("ENV %s:%s" % (k, os.environ[k]))
    print('------------------------------------------------')


def main():
    args = parse_args()
    print_arguments(args)
    print_paddle_envs()
    train_parallel(args)


if __name__ == "__main__":
    main()
