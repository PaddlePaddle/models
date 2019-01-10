from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
import models
import reader
import argparse
import functools
import subprocess
import utils
from utils.learning_rate import cosine_decay
from utility import add_arguments, print_arguments
import models
import models_name

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
add_arg('with_mem_opt',     bool,  True,                 "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('enable_ce',        bool,  False,                "If set True, enable continuous evaluation job.")
add_arg('data_dir',         str,   "./data/ILSVRC2012",  "The ImageNet dataset root dir.")
add_arg('model_category',   str,   "models",             "Whether to use models_name or not, valid value:'models','models_name'" )
add_arg('fp16',             bool,  False,                "Enable half precision training with fp16." )
add_arg('scale_loss',       float, 1.0,                  "Scale loss for fp16." )
# yapf: enable


def set_models(model):
    global models
    if model == "models":
        models = models
    else:
        models = models_name


def optimizer_setting(params):
    ls = params["learning_strategy"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(4e-5))
    elif ls["name"] == "exponential_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size +1)
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        learning_decay_rate_factor=ls["learning_decay_rate_factor"]
        num_epochs_per_decay = ls["num_epochs_per_decay"]
        NUM_GPUS = 1

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate = lr * NUM_GPUS,
                decay_steps = step * num_epochs_per_decay / NUM_GPUS,
                decay_rate = learning_decay_rate_factor),
            momentum=0.9,

            regularization = fluid.regularizer.L2Decay(4e-5))

    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    return optimizer

def net_config(image, label, model, args):
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list,"{} is not lists: {}".format(
        args.model, model_list)

    class_dim = args.class_dim
    model_name = args.model

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
        out = model.net(input=image, class_dim=class_dim)    
        cost, pred = fluid.layers.softmax_with_cross_entropy(out, label, return_softmax=True) 
        if args.scale_loss > 1:
            avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
        else:
            avg_cost = fluid.layers.mean(x=cost)

        acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

    return avg_cost, acc_top1, acc_top5


def cast_fp16_to_fp32(i, o, prog):
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={
            "in_dtype": fluid.core.VarDesc.VarType.FP16,
            "out_dtype": fluid.core.VarDesc.VarType.FP32
        }
    )

def cast_fp32_to_fp16(i, o, prog):
    prog.global_block().append_op(
        type="cast",
        inputs={"X": i},
        outputs={"Out": o},
        attrs={
            "in_dtype": fluid.core.VarDesc.VarType.FP32,
            "out_dtype": fluid.core.VarDesc.VarType.FP16
        }
    )

def copy_to_master_param(p, block):
    v = block.vars.get(p.name, None)
    if v is None:
        raise ValueError("no param name %s found!" % p.name)
    new_p = fluid.framework.Parameter(
        block=block,
        shape=v.shape,
        dtype=fluid.core.VarDesc.VarType.FP32,
        type=v.type,
        lod_level=v.lod_level,
        stop_gradient=p.stop_gradient,
        trainable=p.trainable,
        optimize_attr=p.optimize_attr,
        regularizer=p.regularizer,
        gradient_clip_attr=p.gradient_clip_attr,
        error_clip=p.error_clip,
        name=v.name + ".master")
    return new_p

def update_op_role_var(params_grads, master_params_grads, main_prog):
    orig_grad_name_set = set()
    for _, g in params_grads:
        orig_grad_name_set.add(g.name)
    master_g2p_dict = dict()
    for idx, master in enumerate(master_params_grads):
        orig = params_grads[idx]
        master_g2p_dict[orig[1].name] = [master[0].name, master[1].name]
    for op in main_prog.global_block().ops:
        for oname in op.output_arg_names:
            if oname in orig_grad_name_set:
                # rename
                print("setting to ", master_g2p_dict[oname])
                op._set_attr("op_role_var", master_g2p_dict[oname])

def build_program(is_train, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model_list = [m for m in dir(models) if "__" not in m]
    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    model = models.__dict__[model_name]()
    params_grads = []
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            if args.fp16:
                image = fluid.layers.cast(image, "float16")
            avg_cost, acc_top1, acc_top5 = net_config(image, label, model, args)
            avg_cost.persistable = True
            acc_top1.persistable = True
            acc_top5.persistable = True
            if is_train:
                params = model.params
                params["total_images"] = args.total_images
                params["lr"] = args.lr
                params["num_epochs"] = args.num_epochs
                params["learning_strategy"]["batch_size"] = args.batch_size
                params["learning_strategy"]["name"] = args.lr_strategy

                optimizer = optimizer_setting(params)
                params_grads = optimizer.backward(avg_cost)

                if args.fp16:
                    master_params_grads = []
                    tmp_role = main_prog._current_role
                    OpRole = fluid.core.op_proto_and_checker_maker.OpRole
                    main_prog._current_role = OpRole.Backward
                    for p, g in params_grads:
                        master_param = copy_to_master_param(p, main_prog.global_block())
                        startup_master_param = startup_prog.global_block()._clone_variable(master_param)
                        startup_p = startup_prog.global_block().var(p.name)
                        cast_fp16_to_fp32(startup_p, startup_master_param, startup_prog)

                        if g.name.startswith("batch_norm"):
                            if args.scale_loss > 1:
                                scaled_g = g / float(args.scale_loss)
                            else:
                                scaled_g = g
                            master_params_grads.append([p, scaled_g])
                            continue
                        master_grad = fluid.layers.cast(g, "float32")
                        if args.scale_loss > 1:
                            master_grad = master_grad / float(args.scale_loss)
                        master_params_grads.append([master_param, master_grad])
                    main_prog._current_role = tmp_role

                    optimizer.apply_gradients(master_params_grads)
                    
                    for idx, m_p_g in enumerate(master_params_grads):
                        train_p, train_g = params_grads[idx]
                        if train_p.name.startswith("batch_norm"):
                            continue
                        with main_prog._optimized_guard([m_p_g[0], m_p_g[1]]):
                            cast_fp32_to_fp16(m_p_g[0], train_p, main_prog)
                else:
                    optimizer.minimize(avg_cost)

    return py_reader, avg_cost, acc_top1, acc_top5


def train(args):
    # parameters from arguments
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    model_save_dir = args.model_save_dir

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    if args.enable_ce:
        startup_prog.random_seed = 1000
        train_prog.random_seed = 1000

    train_py_reader, train_cost, train_acc1, train_acc5 = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    test_py_reader, test_cost, test_acc1, test_acc5 = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args)
    test_prog = test_prog.clone(for_test=True)

    if with_memory_optimization:
        fluid.memory_optimize(train_prog)
        fluid.memory_optimize(test_prog)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint, main_program=train_prog)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(['nvidia-smi', '-L']).decode().count('\n')

    train_batch_size = args.batch_size / device_num
    test_batch_size = 8
    if not args.enable_ce:
        train_reader = paddle.batch(
            reader.train(), batch_size=train_batch_size, drop_last=True)
        test_reader = paddle.batch(reader.val(), batch_size=test_batch_size)
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
        test_reader = paddle.batch(
            flowers.test(use_xmap=False), batch_size=test_batch_size)

    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)
    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=bool(args.use_gpu),
        loss_name=train_cost.name)

    train_fetch_list = [train_cost.name, train_acc1.name, train_acc5.name]
    test_fetch_list = [test_cost.name, test_acc1.name, test_acc5.name]

    params = models.__dict__[args.model]().params

    for pass_id in range(params["num_epochs"]):

        train_py_reader.start()

        train_info = [[], [], []]
        test_info = [[], [], []]
        train_time = []
        batch_id = 0
        try:
            while True:
                t1 = time.time()
                loss, acc1, acc5 = train_exe.run(fetch_list=train_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(np.array(loss))
                acc1 = np.mean(np.array(acc1))
                acc5 = np.mean(np.array(acc5))
                train_info[0].append(loss)
                train_info[1].append(acc1)
                train_info[2].append(acc5)
                train_time.append(period)
                if batch_id % 1 == 0:
                    print("Pass {0}, trainbatch {1}, loss {2}, \
                        acc1 {3}, acc5 {4} time {5}"
                          .format(pass_id, batch_id, loss, acc1, acc5,
                                  "%2.2f sec" % period))
                    sys.stdout.flush()
                batch_id += 1
        except fluid.core.EOFException:
            train_py_reader.reset()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        train_speed = np.array(train_time).mean() / train_batch_size

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
                          .format(pass_id, test_batch_id, loss, acc1, acc5,
                                  "%2.2f sec" % period))
                    sys.stdout.flush()
                test_batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, "
              "test_loss {4}, test_acc1 {5}, test_acc5 {6}".format(
                  pass_id, train_loss, train_acc1, train_acc5, test_loss,
                  test_acc1, test_acc5))
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
    models_now = args.model_category
    assert models_now in ["models", "models_name"], "{} is not in lists: {}".format(
            models_now, ["models", "models_name"])
    set_models(models_now)
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()
