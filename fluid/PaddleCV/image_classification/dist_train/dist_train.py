# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import time
import os
import traceback
import functools
import subprocess

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import six
import sys
sys.path.append("..")
import models
import utils
from reader import train, val
from utility import add_arguments, print_arguments
from batch_merge import copyback_repeat_bn_params, append_bn_repeat_init_op
from dist_utils import pserver_prepare, nccl2_prepare
from env import dist_env

def parse_args():
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
    add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
    add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
    add_arg('lr',               float, 0.1,                  "set learning rate.")
    add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
    add_arg('model',            str,   "DistResNet",         "Set the network to use.")
    add_arg('enable_ce',        bool,  False,                "If set True, enable continuous evaluation job.")
    add_arg('data_dir',         str,   "./data/ILSVRC2012",  "The ImageNet dataset root dir.")
    add_arg('model_category',   str,   "models",             "Whether to use models_name or not, valid value:'models','models_name'" )
    add_arg('fp16',             bool,  False,                "Enable half precision training with fp16." )
    add_arg('scale_loss',       float, 1.0,                  "Scale loss for fp16." )
    # for distributed
    add_arg('update_method',      str,  "local",            "Can be local, pserver, nccl2.")
    add_arg('multi_batch_repeat', int,  1,                  "Batch merge repeats.")
    add_arg('start_test_pass',    int,  0,                  "Start test after x passes.")
    add_arg('num_threads',        int,  8,                  "Use num_threads to run the fluid program.")
    add_arg('split_var',          bool, True,               "Split params on pserver.")
    add_arg('async_mode',         bool, False,              "Async distributed training, only for pserver mode.")
    add_arg('reduce_strategy',    str,  "allreduce",        "Choose from reduce or allreduce.")
    add_arg('use_visiontool',       bool,   False,          "Whether to use high performance VisTool reader.")
    add_arg('visiontool_workers',   int,    16,             "Number for visiontool reader workers.") 
    add_arg('freq_drop_scope',      int,    30,             "Number iterations to clean-up the temp variables during execution.")

    # yapf: enable
    args = parser.parse_args()
    return args

def is_mp_mode():
    return True if os.getenv("FLAGS_selected_gpus") else False

def get_device_num():
    if os.getenv("CPU_NUM"):
        return int(os.getenv("CPU_NUM"))
    visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(['nvidia-smi', '-L']).decode().count('\n')
    return device_num

def prepare_reader(is_train, pyreader, args, pass_id=0):
    if is_train:
        reader = train(data_dir=args.data_dir, pass_id_as_seed=pass_id, normalize=False)
    else:
        reader = val(data_dir=args.data_dir, normalize=False)
    if is_train:
        bs = args.batch_size / get_device_num()
    else:
        bs = 16
    pyreader.decorate_paddle_reader(
        paddle.batch(
            reader,
            batch_size=bs))

def prepare_visreader(is_train, pyreader, args):
    import visreader.example.imagenet_demo as imagenet 
    def _parse_kv(r):
        """ parse kv data from sequence file for imagenet
        """
        import cPickle
        k, v = r
        obj = cPickle.loads(v)
        if len(obj['label']) >= 4:
            label = int(obj['label'][3])  # class id
        else:
            label = int(obj['label'][2])
        return obj['image'], label
    pre_maps = [_parse_kv]
    worker_args = {}
    worker_args['cpp_xmap'] = True
    worker_args['use_process'] = False
    worker_args['worker_num'] = args.visiontool_workers
    imagenet.g_settings['worker_args'] = worker_args
    imagenet.g_settings['part_id'] = args.dist_env['trainer_id']
    imagenet.g_settings['part_num'] = args.dist_env['num_trainers']

    if is_train:
        reader = imagenet.train(os.path.join(args.data_dir, "train"), pre_maps=pre_maps)
    else:
        reader = imagenet.val(os.path.join(args.data_dir, "val"), pre_maps=pre_maps)

    batch_size = args.batch_size 
    batch_size_per_gpu = batch_size / get_device_num()
    assert batch_size_per_gpu * get_device_num() == batch_size, \
        "invalid batch_size[%d] for multiple gpus[%d]" % (batch_size, get_device_num())
    pyreader.decorate_paddle_reader(
        paddle.batch(reader, batch_size=batch_size_per_gpu))

def build_program(is_train, main_prog, startup_prog, args):
    pyreader = None
    class_dim = args.class_dim
    image_shape = [int(m) for m in args.image_shape.split(",")]

    trainer_count = args.dist_env["num_trainers"]
    device_num_per_worker = get_device_num()
    with fluid.program_guard(main_prog, startup_prog):
        pyreader = fluid.layers.py_reader(
            capacity=args.batch_size,
            shapes=([-1] + image_shape, (-1, 1)),
            dtypes=('uint8', 'int64'),
            name="train_reader" if is_train else "test_reader",
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(pyreader)
            # normalize image on GPU can achieve better performance
            img_type = "float16" if args.fp16 else "float32"
            cast = fluid.layers.cast(image, img_type)
            img_mean = fluid.layers.create_global_var([3, 1, 1], 0.0, img_type, name="img_mean", persistable=True)
            img_std = fluid.layers.create_global_var([3, 1, 1], 0.0, img_type, name="img_std", persistable=True)
            img_scale = fluid.layers.create_global_var([1], 0.0, img_type, name="img_scale", persistable=True)
            # image = (image / 255.0 - mean) / std
            t1 = fluid.layers.elementwise_sub(cast / img_scale, img_mean, axis=1)
            input = fluid.layers.elementwise_div(t1, img_std, axis=1)
            model_def = models.__dict__[args.model](layers=50, is_train=is_train)
            predict = model_def.net(input, class_dim=class_dim)
            cost, pred = fluid.layers.softmax_with_cross_entropy(predict, label, return_softmax=True) 
            if args.scale_loss > 1:
                avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
            else:
                avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=pred, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=pred, label=label, k=5)

            optimizer = None
            if is_train:
                start_lr = args.lr
                end_lr = args.lr * trainer_count * args.multi_batch_repeat
                if os.getenv("FLAGS_selected_gpus"):
                    # in multi process mode, "trainer_count" will be total devices
                    # in the whole cluster, and we need to scale num_of nodes.
                    end_lr /= device_num_per_worker

                total_images = args.total_images / trainer_count
                step = int(total_images / (args.batch_size * args.multi_batch_repeat) + 1)
                warmup_steps = step * 5  # warmup 5 passes
                epochs = [30, 60, 80]
                bd = [step * e for e in epochs]
                base_lr = end_lr
                lr = []
                lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
                print("start lr: %s, end lr: %s, decay boundaries: %s" % (
                    start_lr,
                    end_lr,
                    bd
                ))

                # NOTE: we put weight decay in layers config, and remove
                # weight decay on bn layers, so don't add weight decay in
                # optimizer config.
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=utils.learning_rate.lr_warmup(
                        fluid.layers.piecewise_decay(
                            boundaries=bd, values=lr),
                        warmup_steps, start_lr, end_lr),
                    momentum=0.9)
                if args.fp16:
                    params_grads = optimizer.backward(avg_cost)
                    master_params_grads = utils.create_master_params_grads(
                        params_grads, main_prog, startup_prog, args.scale_loss)
                    optimizer.apply_gradients(master_params_grads)
                    utils.master_param_to_train_param(master_params_grads, params_grads, main_prog)
                else:
                    optimizer.minimize(avg_cost)

    # prepare reader for current program
    if args.use_visiontool:
        prepare_visreader(is_train, pyreader, args)
    else:
        prepare_reader(is_train, pyreader, args)

    return pyreader, avg_cost, batch_acc1, batch_acc5


def test_single(exe, test_prog, args, pyreader, fetch_list):
    acc1 = fluid.metrics.Accuracy()
    acc5 = fluid.metrics.Accuracy()
    test_losses = []
    pyreader.start()
    while True:
        try:
            acc_rets = exe.run(program=test_prog, fetch_list=fetch_list)
            test_losses.append(acc_rets[0])
            acc1.update(value=np.array(acc_rets[1]), weight=args.batch_size)
            acc5.update(value=np.array(acc_rets[2]), weight=args.batch_size)
        except fluid.core.EOFException:
            pyreader.reset()
            break
    test_avg_loss = np.mean(np.array(test_losses))
    return test_avg_loss, np.mean(acc1.eval()), np.mean(acc5.eval())

def run_pserver(train_prog, startup_prog):
    server_exe = fluid.Executor(fluid.CPUPlace())
    server_exe.run(startup_prog)
    server_exe.run(train_prog)

def train_parallel(args):
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    train_pyreader, train_cost, train_acc1, train_acc5 = build_program(True, train_prog, startup_prog, args)
    test_pyreader, test_cost, test_acc1, test_acc5 = build_program(False, test_prog, startup_prog, args)

    if args.update_method == "pserver":
        train_prog, startup_prog = pserver_prepare(args, train_prog, startup_prog)
    elif args.update_method == "nccl2":
        nccl2_prepare(args, startup_prog)

    if args.dist_env["training_role"] == "PSERVER":
        run_pserver(train_prog, startup_prog)
        exit(0)

    if args.use_gpu:
        # NOTE: for multi process mode: one process per GPU device.        
        gpu_id = 0
        if os.getenv("FLAGS_selected_gpus"):
            gpu_id = int(os.getenv("FLAGS_selected_gpus"))
    place = core.CUDAPlace(gpu_id) if args.use_gpu else core.CPUPlace()

    startup_exe = fluid.Executor(place)
    if args.multi_batch_repeat > 1:
        append_bn_repeat_init_op(train_prog, startup_prog, args.multi_batch_repeat)
    startup_exe.run(startup_prog)

    np_tensors = {}
    np_tensors["img_mean"] = np.array([0.485, 0.456, 0.406]).astype("float16" if args.fp16 else "float32").reshape((3, 1, 1))
    np_tensors["img_std"] = np.array([0.229, 0.224, 0.225]).astype("float16" if args.fp16 else "float32").reshape((3, 1, 1))
    np_tensors["img_scale"] = np.array([255.0]).astype("float16" if args.fp16 else "float32")
    for vname, np_tensor in np_tensors.items():
        var = fluid.global_scope().find_var(vname)
        if args.fp16:
            var.get_tensor().set(np_tensor.view(np.uint16), place)
        else:
            var.get_tensor().set(np_tensor, place)


    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = args.num_threads
    strategy.num_iteration_per_drop_scope = args.freq_drop_scopes
    build_strategy = fluid.BuildStrategy()
    if args.multi_batch_repeat > 1:
        pass_builder = build_strategy._finalize_strategy_and_create_passes()
        mypass = pass_builder.insert_pass(
            len(pass_builder.all_passes()) - 2, "multi_batch_merge_pass")
        mypass.set_int("num_repeats", args.multi_batch_repeat)
    if args.reduce_strategy == "reduce":
        build_strategy.reduce_strategy = fluid.BuildStrategy(
        ).ReduceStrategy.Reduce
    else:
        build_strategy.reduce_strategy = fluid.BuildStrategy(
        ).ReduceStrategy.AllReduce

    if args.update_method == "pserver" or args.update_method == "local":
        # parameter server mode distributed training, merge
        # gradients on local server, do not initialize
        # ParallelExecutor with multi server all-reduce mode.
        num_trainers = 1
        trainer_id = 0
    else:
        num_trainers = args.dist_env["num_trainers"]
        trainer_id = args.dist_env["trainer_id"]

    exe = fluid.ParallelExecutor(
        True,
        train_cost.name,
        main_program=train_prog,
        exec_strategy=strategy,
        build_strategy=build_strategy,
        num_trainers=num_trainers,
        trainer_id=trainer_id)

    over_all_start = time.time()
    fetch_list = [train_cost.name, train_acc1.name, train_acc5.name]
    # for multi-processes mode, one GPU device per worker
    # for single-process mode, multiple GPU devices per worker
    batch_size_per_worker = args.batch_size / get_device_num() if is_mp_mode() else args.batch_size
    steps_per_pass = args.total_images / args.dist_env["num_trainers"] / batch_size_per_worker
    for pass_id in range(args.num_epochs):
        num_samples = 0
        start_time = time.time()
        if args.use_visiontool:
            # vistool reader is ulimited data, don't need restart pyreader.
            # And should be aware that vistool readre doesn't support global shuffle.
            if pass_id == 0: train_pyreader.start()
        else:
            # use pass_id+1 as per pass global shuffle for distributed training
            prepare_reader(True, train_pyreader, args, pass_id + 1)
            train_pyreader.start()
        batch_id = 0
        while True:
            batch_id += 1
            try:
                if batch_id % 30 == 0:
                    fetch_ret = exe.run(fetch_list)
                    fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                    print("Pass %d, batch [%d/%d], loss %s, acc1: %s, acc5: %s, avg batch time %.4f" %
                        (pass_id, batch_id, steps_per_pass, fetched_data[0], fetched_data[1],
                         fetched_data[2], (time.time()-start_time) / batch_id))
                else:
                    fetch_ret = exe.run([])
            except fluid.core.EOFException:
                break
            except fluid.core.EnforceNotMet:
                traceback.print_exc()
                break
            num_samples += args.batch_size
            if batch_id > steps_per_pass and args.use_visiontool:
                break

        print_train_time(start_time, time.time(), num_samples)
        if (args.use_visiontool and pass_id == args.num_epochs - 1) or not args.use_visiontool:
            train_pyreader.reset()

        if pass_id >= args.start_test_pass:
            if args.multi_batch_repeat > 1:
                copyback_repeat_bn_params(train_prog)
            test_fetch_list = [test_cost.name, test_acc1.name, test_acc5.name]
            test_ret = test_single(startup_exe, test_prog, args, test_pyreader,test_fetch_list)
            print("Pass: %d, Test Loss %s, test acc1: %s, test acc5: %s\n" %
                  (pass_id, test_ret[0], test_ret[1], test_ret[2]))
    # TODO(Yancey1989): need to fix on 
    if args.update_method == "pserver":
        startup_exe.close()
    print("total train time: ", time.time() - over_all_start)


def print_train_time(start_time, end_time, num_samples):
    train_elapsed = end_time - start_time
    examples_per_sec = num_samples / train_elapsed
    print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
          (num_samples, train_elapsed, examples_per_sec))


def print_paddle_envs():
    print('----------- Configuration envs -----------')
    for k in os.environ:
        if "PADDLE_" in k:
            print("ENV %s:%s" % (k, os.environ[k]))
    print('------------------------------------------------')


def main():
    args = parse_args()
    print_arguments(args)
    print_paddle_envs()
    args.dist_env = dist_env()
    train_parallel(args)

if __name__ == "__main__":
    main()
