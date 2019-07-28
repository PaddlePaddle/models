#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import sys
import functools
import math


def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags({
    'FLAGS_eager_delete_tensor_gb': 0,  # enable gc 
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})
import argparse
import functools
import subprocess
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
import reader_cv2 as reader
import utils
import models
from utils.fp16_utils import create_master_params_grads, master_param_to_train_param
from utils.utility import add_arguments, print_arguments, check_gpu
from utils.learning_rate import cosine_decay_with_warmup
from dist_train import dist_utils

IMAGENET1000 = 1281167
num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('batch_size',       int,   256,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   1281167,              "Training image number.")
add_arg('num_epochs',       int,   120,                  "number of epochs.")
add_arg('class_dim',        int,   1000,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('model_save_dir',   str,   "output",             "model save directory")
add_arg('with_mem_opt',     bool,  False,                 "Whether to use memory optimization or not.")
add_arg('with_inplace',     bool,  True,                 "Whether to use inplace memory optimization.")
add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('enable_ce',        bool,  False,                "If set True, enable continuous evaluation job.")
add_arg('data_dir',         str,   "./data/ILSVRC2012/",  "The ImageNet dataset root dir.")
add_arg('fp16',             bool,  False,                "Enable half precision training with fp16." )
add_arg('scale_loss',       float, 1.0,                  "Scale loss for fp16." )
add_arg('l2_decay',         float, 1e-4,                 "L2_decay parameter.")
add_arg('momentum_rate',    float, 0.9,                  "momentum_rate.")
add_arg('use_label_smoothing',      bool,      False,        "Whether to use label_smoothing or not")
add_arg('label_smoothing_epsilon',      float,     0.2,      "Set the label_smoothing_epsilon parameter")
add_arg('lower_scale',      float,     0.08,      "Set the lower_scale in ramdom_crop")
add_arg('lower_ratio',      float,     3./4.,      "Set the lower_ratio in ramdom_crop")
add_arg('upper_ratio',      float,     4./3.,      "Set the upper_ratio in ramdom_crop")
add_arg('resize_short_size',      int,     256,      "Set the resize_short_size")
add_arg('use_mixup',      bool,      False,        "Whether to use mixup or not")
add_arg('mixup_alpha',      float,     0.2,      "Set the mixup_alpha parameter")
add_arg('is_distill',       bool,  False,        "is distill or not")

def optimizer_setting(params):
    ls = params["learning_strategy"]
    l2_decay = params["l2_decay"]
    momentum_rate = params["momentum_rate"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(math.ceil(float(total_images) / batch_size))
        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))

    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        step = int(math.ceil(float(total_images) / batch_size))
        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))

    elif ls["name"] == "cosine_warmup_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        step = int(math.ceil(float(total_images) / batch_size))
        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay_with_warmup(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))

    elif ls["name"] == "linear_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        num_epochs = params["num_epochs"]
        start_lr = params["lr"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        end_lr = 0
        total_step = int((total_images / batch_size) * num_epochs)
        lr = fluid.layers.polynomial_decay(
            start_lr, total_step, end_lr, power=1)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "adam":
        lr = params["lr"]
        optimizer = fluid.optimizer.Adam(learning_rate=lr)
    elif ls["name"] == "rmsprop_cosine":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        step = int(math.ceil(float(total_images) / batch_size))
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = fluid.optimizer.RMSProp(
            learning_rate=fluid.layers.cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay),
            # RMSProp Optimizer: Apply epsilon=1 on ImageNet.
            epsilon=1
        )
    else:
        lr = params["lr"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))

    return optimizer

def calc_loss(epsilon,label,class_dim,softmax_out,use_label_smoothing):
    if use_label_smoothing:
        label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
        smooth_label = fluid.layers.label_smooth(label=label_one_hot, epsilon=epsilon, dtype="float32")
        loss = fluid.layers.cross_entropy(input=softmax_out, label=smooth_label, soft_label=True)
    else:
        loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
    return loss


def net_config(image, model, args, is_train, label=0, y_a=0, y_b=0, lam=0.0):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not lists: {}".format(args.model,
                                                                  model_list)
    class_dim = args.class_dim
    model_name = args.model
    use_mixup = args.use_mixup
    use_label_smoothing = args.use_label_smoothing
    epsilon = args.label_smoothing_epsilon

    if args.enable_ce:
        assert model_name == "SE_ResNeXt50_32x4d"
        model.params["dropout_seed"] = 100
        class_dim = 102

    if model_name == "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)

    else:
        if not args.is_distill:
            out = model.net(input=image, class_dim=class_dim)
            softmax_out = fluid.layers.softmax(out, use_cudnn=False)
            if is_train:
                if use_mixup:
                    loss_a = calc_loss(epsilon,y_a,class_dim,softmax_out,use_label_smoothing)
                    loss_b = calc_loss(epsilon,y_b,class_dim,softmax_out,use_label_smoothing)
                    loss_a_mean = fluid.layers.mean(x = loss_a)
                    loss_b_mean = fluid.layers.mean(x = loss_b)
                    cost = lam * loss_a_mean + (1 - lam) * loss_b_mean
                    avg_cost = fluid.layers.mean(x=cost)
                    if args.scale_loss > 1:
                        avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
                    return avg_cost
                else:
                    cost = calc_loss(epsilon,label,class_dim,softmax_out,use_label_smoothing)

            else:
                cost = fluid.layers.cross_entropy(input=softmax_out, label=label)
        else:
            out1, out2 = model.net(input=image, class_dim=args.class_dim)
            softmax_out1, softmax_out = fluid.layers.softmax(out1), fluid.layers.softmax(out2)
            smooth_out1 = fluid.layers.label_smooth(label=softmax_out1, epsilon=0.0, dtype="float32")
            cost = fluid.layers.cross_entropy(input=softmax_out, label=smooth_out1, soft_label=True)

        avg_cost = fluid.layers.mean(cost)
        if args.scale_loss > 1:
            avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
        acc_top1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=softmax_out, label=label, k=5)

    return avg_cost, acc_top1, acc_top5

def build_program(is_train, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    model = models.__dict__[model_name]()
    with fluid.program_guard(main_prog, startup_prog):
        use_mixup = args.use_mixup
        if is_train and use_mixup:
            py_reader = fluid.layers.py_reader(
                capacity=16,
                shapes=[[-1] + image_shape, [-1, 1], [-1, 1], [-1, 1]],
                lod_levels=[0, 0, 0, 0],
                dtypes=["float32", "int64", "int64", "float32"],
                use_double_buffer=True)
        else:
            py_reader = fluid.layers.py_reader(
                capacity=16,
                shapes=[[-1] + image_shape, [-1, 1]],
                lod_levels=[0, 0],
                dtypes=["float32", "int64"],
                use_double_buffer=True)

        with fluid.unique_name.guard():
            if is_train and  use_mixup:
                image, y_a, y_b, lam = fluid.layers.read_file(py_reader)
                if args.fp16:
                    image = fluid.layers.cast(image, "float16")
                avg_cost = net_config(image=image, y_a=y_a, y_b=y_b, lam=lam, model=model, args=args, label=0, is_train=True)
                avg_cost.persistable = True
                build_program_out = [py_reader, avg_cost]
            else:
                image, label = fluid.layers.read_file(py_reader)
                if args.fp16:
                    image = fluid.layers.cast(image, "float16")
                avg_cost, acc_top1, acc_top5 = net_config(image, model, args, label=label, is_train=is_train)
                avg_cost.persistable = True
                acc_top1.persistable = True
                acc_top5.persistable = True
                build_program_out = [py_reader, avg_cost, acc_top1, acc_top5]

            if is_train:
                params = model.params
                params["total_images"] = args.total_images
                params["lr"] = args.lr
                params["num_epochs"] = args.num_epochs
                params["learning_strategy"]["batch_size"] = args.batch_size
                params["learning_strategy"]["name"] = args.lr_strategy
                params["l2_decay"] = args.l2_decay
                params["momentum_rate"] = args.momentum_rate

                optimizer = optimizer_setting(params)
                if args.fp16:
                    params_grads = optimizer.backward(avg_cost)
                    master_params_grads = create_master_params_grads(
                        params_grads, main_prog, startup_prog, args.scale_loss)
                    optimizer.apply_gradients(master_params_grads)
                    master_param_to_train_param(master_params_grads,
                                                params_grads, main_prog)
                else:
                    optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()
                global_lr.persistable=True
                build_program_out.append(global_lr)

    return build_program_out

def get_device_num():
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1:
        return 1
    return fluid.core.get_cuda_device_count()

def train(args):
    # parameters from arguments
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    model_save_dir = args.model_save_dir
    use_mixup = args.use_mixup

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    if args.enable_ce:
        startup_prog.random_seed = 1000
        train_prog.random_seed = 1000

    b_out = build_program(
                     is_train=True,
                     main_prog=train_prog,
                     startup_prog=startup_prog,
                     args=args)
    if use_mixup:
        train_py_reader, train_cost, global_lr = b_out[0], b_out[1], b_out[2]
        train_fetch_vars = [train_cost, global_lr]
        train_fetch_list = []
        for var in train_fetch_vars:
            var.persistable=True
            train_fetch_list.append(var.name)

    else:
        train_py_reader, train_cost, train_acc1, train_acc5, global_lr = b_out[0],b_out[1],b_out[2],b_out[3],b_out[4]
        train_fetch_vars = [train_cost, train_acc1, train_acc5, global_lr]
        train_fetch_list = []
        for var in train_fetch_vars:
            var.persistable=True
            train_fetch_list.append(var.name)

    b_out_test = build_program(
                     is_train=False,
                     main_prog=test_prog,
                     startup_prog=startup_prog,
                     args=args)
    test_py_reader, test_cost, test_acc1, test_acc5 = b_out_test[0],b_out_test[1],b_out_test[2],b_out_test[3]
    test_prog = test_prog.clone(for_test=True)

    if with_memory_optimization:
        fluid.memory_optimize(train_prog)
        fluid.memory_optimize(test_prog)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint, main_program=train_prog)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    if args.use_gpu:
        device_num = get_device_num()
    else:
        device_num = 1
    train_batch_size = args.batch_size / device_num

    test_batch_size = 16
    if not args.enable_ce:
        # NOTE: the order of batch data generated by batch_reader
        # must be the same in the respective processes.
        shuffle_seed = 1 if num_trainers > 1 else None
        train_reader = reader.train(settings=args, batch_size=train_batch_size, shuffle_seed=shuffle_seed)
        test_reader = reader.val(settings=args, batch_size=test_batch_size)
    else:
        # use flowers dataset for CE and set use_xmap False to avoid disorder data
        # but it is time consuming. For faster speed, need another dataset.
        import random
        random.seed(0)
        np.random.seed(0)
        train_reader = paddle.batch(
            flowers.train(use_xmap=False),
            batch_size=train_batch_size,
            drop_last=True)
        if num_trainers > 1:
            train_reader = fluid.contrib.reader.distributed_batch_reader(train_reader)
        test_reader = paddle.batch(
            flowers.test(use_xmap=False), batch_size=test_batch_size)

    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)


    test_fetch_vars = [test_cost, test_acc1, test_acc5]
    test_fetch_list = []
    for var in test_fetch_vars:
        var.persistable=True
        test_fetch_list.append(var.name)

    # use_ngraph is for CPU only, please refer to README_ngraph.md for details
    use_ngraph = os.getenv('FLAGS_use_ngraph')
    if not use_ngraph:
        build_strategy = fluid.BuildStrategy()
        # memopt may affect GC results
        #build_strategy.memory_optimize = args.with_mem_opt
        build_strategy.enable_inplace = args.with_inplace
        #build_strategy.fuse_all_reduce_ops=1

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = device_num
        exec_strategy.num_iteration_per_drop_scope = 10
        if num_trainers > 1 and args.use_gpu:
            dist_utils.prepare_for_multi_process(exe, build_strategy, train_prog)
            # NOTE: the process is fast when num_threads is 1
            # for multi-process training.
            exec_strategy.num_threads = 1

        train_exe = fluid.ParallelExecutor(
            main_program=train_prog,
            use_cuda=bool(args.use_gpu),
            loss_name=train_cost.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    else:
        train_exe = exe

    params = models.__dict__[args.model]().params
    for pass_id in range(params["num_epochs"]):

        train_py_reader.start()
        train_info = [[], [], []]
        test_info = [[], [], []]
        train_time = []
        batch_id = 0
        time_record=[]
        try:
            while True:
                t1 = time.time()
                if use_mixup:
                    if use_ngraph:
                        loss, lr = train_exe.run(train_prog, fetch_list=train_fetch_list)
                    else:
                        loss, lr = train_exe.run(fetch_list=train_fetch_list)
                else:
                    if use_ngraph:
                        loss, acc1, acc5, lr = train_exe.run(train_prog, fetch_list=train_fetch_list)
                    else:
                        loss, acc1, acc5, lr = train_exe.run(fetch_list=train_fetch_list)

                    acc1 = np.mean(np.array(acc1))
                    acc5 = np.mean(np.array(acc5))
                    train_info[1].append(acc1)
                    train_info[2].append(acc5)

                t2 = time.time()
                period = t2 - t1
                time_record.append(period)

                loss = np.mean(np.array(loss))
                train_info[0].append(loss)
                lr = np.mean(np.array(lr))
                train_time.append(period)

                if batch_id % 10 == 0:
                    period = np.mean(time_record)
                    time_record=[]
                    if use_mixup:
                        print("Pass {0}, trainbatch {1}, loss {2}, lr {3}, time {4}"
                              .format(pass_id, batch_id, "%.5f"%loss, "%.5f" %lr, "%2.2f sec" % period))
                    else:
                        print("Pass {0}, trainbatch {1}, loss {2}, \
                            acc1 {3}, acc5 {4}, lr {5}, time {6}"
                              .format(pass_id, batch_id, "%.5f"%loss, "%.5f"%acc1, "%.5f"%acc5, "%.5f" %
                                      lr, "%2.2f sec" % period))
                    sys.stdout.flush()
                batch_id += 1
        except fluid.core.EOFException:
            train_py_reader.reset()

        train_loss = np.array(train_info[0]).mean()
        if not use_mixup:
            train_acc1 = np.array(train_info[1]).mean()
            train_acc5 = np.array(train_info[2]).mean()
        train_speed = np.array(train_time).mean() / (train_batch_size *
                                                     device_num)

        test_py_reader.start()

        test_batch_id = 0
        try:
            while True:
                t1 = time.time()
                loss, acc1, acc5 = exe.run(program=test_prog,
                                           fetch_list=test_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(loss)
                acc1 = np.mean(acc1)
                acc5 = np.mean(acc5)
                test_info[0].append(loss)
                test_info[1].append(acc1)
                test_info[2].append(acc5)
                if test_batch_id % 10 == 0:
                    print("Pass {0},testbatch {1},loss {2}, \
                        acc1 {3},acc5 {4},time {5}"
                          .format(pass_id, test_batch_id, "%.5f"%loss,"%.5f"%acc1, "%.5f"%acc5,
                                  "%2.2f sec" % period))
                    sys.stdout.flush()
                test_batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        if use_mixup:
            print("End pass {0}, train_loss {1}, test_loss {2}, test_acc1 {3}, test_acc5 {4}".format(
                      pass_id, "%.5f"%train_loss, "%.5f"%test_loss, "%.5f"%test_acc1, "%.5f"%test_acc5))
        else:

            print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, "
                  "test_loss {4}, test_acc1 {5}, test_acc5 {6}".format(
                      pass_id, "%.5f"%train_loss, "%.5f"%train_acc1, "%.5f"%train_acc5, "%.5f"%test_loss,
                      "%.5f"%test_acc1, "%.5f"%test_acc5))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir + '/' + model_name,
                                  str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path, main_program=train_prog)

        # This is for continuous evaluation only
        if args.enable_ce and pass_id == args.num_epochs - 1:
            if device_num == 1:
                # Use the mean cost/acc for training
                print("kpis	train_cost	%s" % train_loss)
                print("kpis	train_acc_top1	%s" % train_acc1)
                print("kpis	train_acc_top5	%s" % train_acc5)
                # Use the mean cost/acc for testing
                print("kpis	test_cost	%s" % test_loss)
                print("kpis	test_acc_top1	%s" % test_acc1)
                print("kpis	test_acc_top5	%s" % test_acc5)
                print("kpis	train_speed	%s" % train_speed)
            else:
                # Use the mean cost/acc for training
                print("kpis	train_cost_card%s	%s" % (device_num, train_loss))
                print("kpis	train_acc_top1_card%s	%s" %
                      (device_num, train_acc1))
                print("kpis	train_acc_top5_card%s	%s" %
                      (device_num, train_acc5))
                # Use the mean cost/acc for testing
                print("kpis	test_cost_card%s	%s" % (device_num, test_loss))
                print("kpis	test_acc_top1_card%s	%s" % (device_num, test_acc1))
                print("kpis	test_acc_top5_card%s	%s" % (device_num, test_acc5))
                print("kpis	train_speed_card%s	%s" % (device_num, train_speed))


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    train(args)


if __name__ == '__main__':
    main()
