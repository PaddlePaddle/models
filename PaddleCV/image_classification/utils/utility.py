"""Contains common utility functions."""
#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import distutils.util
import numpy as np
import six
import argparse
import functools
import logging
import sys
import os
import warnings

import paddle
import paddle.fluid as fluid


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-------------  Configuration Arguments -------------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%25s : %s" % (arg, value))
    print("----------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's arg ument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def parse_args():
    """Add arguments

    Returns: 
        all args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable

    # ENV
    add_arg('use_gpu',                  bool,   True,                   "Whether to use GPU.")
    add_arg('model_save_dir',           str,    "./data/output",        "The directory path to save model.")
    add_arg('data_dir',                 str,    "./data/ILSVRC2012/",   "The ImageNet dataset root directory.")
    add_arg('pretrained_model',         str,    None,                   "Whether to load pretrained model.")
    add_arg('checkpoint',               str,    None,                   "Whether to resume checkpoint.")

    add_arg('print_step',               int,    10,                     "The steps interval to print logs")
    add_arg('save_step',                int,    100,                    "The steps interval to save checkpoints")

    # SOLVER AND HYPERPARAMETERS
    add_arg('model',                    str,    "AlexNet",   "The name of network.")
    add_arg('total_images',             int,    1281167,                "The number of total training images.")
    add_arg('num_epochs',               int,    120,                    "The number of total epochs.")
    add_arg('class_dim',                int,    1000,                   "The number of total classes.")
    add_arg('image_shape',              str,    "3,224,224",            "The size of Input image, order: [channels, height, weidth] ")
    add_arg('batch_size',               int,    8,                    "Minibatch size on all devices.")
    add_arg('test_batch_size',          int,    16,                     "Test batch size.")
    add_arg('lr',                       float,  0.1,                    "The learning rate.")
    add_arg('lr_strategy',              str,    "piecewise_decay",      "The learning rate decay strategy.")
    add_arg('l2_decay',                 float,  1e-4,                   "The l2_decay parameter.")
    add_arg('momentum_rate',            float,  0.9,                    "The value of momentum_rate.")
    #add_arg('step_epochs',              nargs-int-type,     [30, 60, 90]  "piecewise decay step")
    # READER AND PREPROCESS
    add_arg('lower_scale',              float,  0.08,                   "The value of lower_scale in ramdom_crop")
    add_arg('lower_ratio',              float,  3./4.,                  "The value of lower_ratio in ramdom_crop")
    add_arg('upper_ratio',              float,  4./3.,                  "The value of upper_ratio in ramdom_crop")
    add_arg('resize_short_size',        int,    256,                    "The value of resize_short_size")
    add_arg('use_mixup',                bool,   False,                  "Whether to use mixup")
    add_arg('mixup_alpha',              float,  0.2,                    "The value of mixup_alpha")

    # SWITCH
    #add_arg('use_mem_opt',              bool,   False,                  "Whether to use memory optimization.")
    add_arg('use_inplace',              bool,   True,                   "Whether to use inplace memory optimization.")
    add_arg('enable_ce',                bool,   False,                  "Whether to enable continuous evaluation job.")
    add_arg('use_fp16',                 bool,   False,                  "Whether to enable half precision training with fp16." )
    add_arg('scale_loss',               float,  1.0,                    "The value of scale_loss for fp16." )
    add_arg('use_label_smoothing',      bool,   False,                  "Whether to use label_smoothing")
    add_arg('label_smoothing_epsilon',  float,  0.2,                    "The value of label_smoothing_epsilon parameter")
    #add_arg('use_distill',              bool,   False,                  "Whether to use distill")
    add_arg('random_seed',              int,    1000,                   "random seed")
    # yapf: enable
    args = parser.parse_args()
    return args


def check_args(args):
    """check arguments before running 
    """

    # check models name
    sys.path.append("..")
    import models
    model_list = [m for m in dir(models) if "__" not in m]
    assert args.model in model_list, "{} is not in lists: {}, please check the model name".format(
        args.model, model_list)

    # check learning rate strategy
    lr_strategy_list = [
        "piecewise_decay", "cosine_decay", "linear_decay", "cosine_decay_warmup"
    ]
    if args.lr_strategy not in lr_strategy_list:
        warnings.warn(
            "{} is not in lists: {}, Use default learning strategy now".format(
                args.lr_strategy, lr_strategy_list))

    if args.model == "GooLeNet":
        assert arg.use_mixup == True, "Cannot use mixup processing in GoogLeNet, please set use_mixup = False"

    if args.pretrained_model is not None:
        assert os.path.isdir(args.pretrained_model)

    if args.checkpoint is not None:
        assert os.path.isdir(args.checkpoint)
        # when use gpu, the number of visible gpu should divide batch size
    assert args.batch_size % fluid.core.get_cuda_device_count() == 0

    def check_gpu():
        """ 
        Log error and exit when set use_gpu=true in paddlepaddle
        cpu version.
        """
        logger = logging.getLogger(__name__)
        err = "Config use_gpu cannot be set as true while you are " \
                "using paddlepaddle cpu version ! \nPlease try: \n" \
                "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
                "\t2. Set use_gpu as false in config file to run " \
                "model on CPU"

        try:
            if args.use_gpu and not fluid.is_compiled_with_cuda():
                print(err)
                sys.exit(1)
        except Exception as e:
            pass

    check_gpu()
    #temporary disable:
    if args.enable_ce == True:
        raise


def get_device_num():
    """Obtain the num ber of available GPU cards

    Returns:
        the num of devices
    """

    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1: return 1
    visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(
            ['nvidia-smi', '-L']).decode().count('\n')
    print("...Running on ", device_num, " GPU cards")
    return device_num


def init_from(exe, args, program):

    if args.checkpoint is not None:
        fluid.io.load_persistables(exe, args.checkpoint, main_program=program)

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(
            exe,
            args.pretrained_model,
            main_program=program,
            predicate=if_exist)


def init_from_checkpoint(args, exe, program):

    assert isinstance(args.init_from_checkpoint, str)

    if not os.path.exists(args.init_from_checkpoint):
        raise Warning("the checkpoint path %s does not exist." %
                      args.init_from_checkpoint)
        return False

    fluid.io.load_persistables(
        executor=exe,
        dirname=args.init_from_checkpoint,
        main_program=program,
        filename="checkpoint.pdckpt")

    print("finish init model from checkpoint at %s" %
          (args.init_from_checkpoint))

    return True


def save_checkpoint(args, exe, program, pass_id):

    assert isinstance(args.model_save_dir, str)

    checkpoint_path = os.path.join(args.model_save_dir, args.model,
                                   str(pass_id))

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    fluid.io.save_persistables(
        exe,
        checkpoint_path,
        main_program=program,
        filename="checkpoint.pdckpt")

    print("save checkpoint at %s" % (checkpoint_path))
    return True


def create_pyreader(is_train, args):
    """  
    use PyReader
    """
    image_shape = [int(m) for m in args.image_shape.split(",")]

    feed_image = fluid.layers.data(
        name="feed_image", shape=image_shape, dtype="float32", lod_level=0)

    feed_label = fluid.layers.data(
        name="feed_label", shape=[1], dtype="int64", lod_level=0)
    feed_y_a = fluid.layers.data(
        name="feed_y_a", shape=[1], dtype="int64", lod_level=0)
    feed_y_b = fluid.layers.data(
        name="feed_y_b", shape=[1], dtype="int64", lod_level=0)
    feed_lam = fluid.layers.data(
        name="feed_lam", shape=[1], dtype="float32", lod_level=0)

    if is_train and args.use_mixup:  # and args.model != "GoogLeNet":
        print("========Create pyreader========")
        py_reader = fluid.io.PyReader(
            feed_list=[feed_image, feed_y_a, feed_y_b, feed_lam],
            capacity=64,
            use_double_buffer=True,
            iterable=False)
        return py_reader, [feed_image, feed_y_a, feed_y_b, feed_lam]
    else:
        py_reader = fluid.io.PyReader(
            feed_list=[feed_image, feed_label],
            capacity=64,
            use_double_buffer=True,
            iterable=False)

        return py_reader, [feed_image, feed_label]


def print_info(pass_id, batch_id, print_step, metrics, time_info, info_mode):
    if info_mode == "batch":
        if batch_id % print_step == 0:
            #if isinstance(metrics,np.ndarray):
            if len(metrics) == 2:
                loss, lr = metrics
                print(
                    "[Pass {0}, train batch {1}], loss {2}, lr {3}, elapse {4}".
                    format(pass_id, batch_id, "%.5f" % loss, "%.5f" % lr,
                           "%2.2f sec" % time_info))
            # no mixup putput
            elif len(metrics) == 4:
                loss, acc1, acc5, lr = metrics
                print(
                    "[Pass {0}, train batch {1}], loss {2}, acc1 {3}, acc5 {4}, lr {5}, elapse {6}".
                    format(pass_id, batch_id, "%.5f" % loss, "%.5f" % acc1,
                           "%.5f" % acc5, "%.5f" % lr, "%2.2f sec" % time_info))
        # test output
            elif len(metrics) == 3:
                loss, acc1, acc5 = metrics
                print(
                    "[Pass {0}, test batch {1}], loss {2}, acc1 {3}, acc5 {4}, elapse {5}".
                    format(pass_id, batch_id, "%.5f" % loss, "%.5f" % acc1,
                           "%.5f" % acc5, "%2.2f sec" % time_info))
            else:
                print(
                    "length of metrics is not implenmented, It maybe cause by wrong format of build_program_output!!"
                )
            sys.stdout.flush()

    elif info_mode == "epoch":
        ## TODO add time elapse
        #if isinstance(metrics,np.ndarray):
        if len(metrics) == 5:
            train_loss, _, test_loss, test_acc1, test_acc5 = metrics
            print(
                "[End pass {0}], train_loss {1}, test_loss {2}, test_acc1 {3}, test_acc5 {4}".
                format(pass_id, "%.5f" % train_loss, "%.5f" % test_loss, "%.5f"
                       % test_acc1, "%.5f" % test_acc5))
        elif len(metrics) == 7:
            train_loss, train_acc1, train_acc5, _, test_loss, test_acc1, test_acc5 = metrics
            print(
                "[End pass {0}], train_loss {1}, train_acc1 {2}, train_acc5 {3},test_loss {4}, test_acc1 {5}, test_acc5 {6}".
                format(pass_id, "%.5f" % train_loss, "%.5f" % train_acc1, "%.5f"
                       % train_acc5, "%.5f    " % test_loss, "%.5f" % test_acc1,
                       "%.5f" % test_acc5))
        sys.stdout.flush()
    elif info_mode == "ce":
        print("CE TESTING CODE IS HERE")
    else:
        print("illegal info_mode!!!")


def best_strategy(args, program, loss):
    # use_ngraph is for CPU only, please refer to README_ngraph.md for details
    use_ngraph = os.getenv('FLAGS_use_ngraph')
    if not use_ngraph:
        build_strategy = fluid.BuildStrategy()
        # memopt may affect GC results
        #build_strategy.memory_optimize = args.with_mem_opt
        build_strategy.enable_inplace = args.use_inplace
        #build_strategy.fuse_all_reduce_ops=1

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = fluid.core.get_cuda_device_count()
        exec_strategy.num_iteration_per_drop_scope = 10

        num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

        if num_trainers > 1 and args.use_gpu:
            dist_utils.prepare_for_multi_process(exe, build_strategy,
                                                 train_prog)
            # NOTE: the process is fast when num_threads is 1
            # for multi-process training.
            exec_strategy.num_threads = 1

        train_exe = fluid.ParallelExecutor(
            main_program=program,
            use_cuda=bool(args.use_gpu),
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    else:
        train_exe = exe
    print("[Program is running on ", fluid.core.get_cuda_device_count(),
          " cards ]")
    return train_exe
