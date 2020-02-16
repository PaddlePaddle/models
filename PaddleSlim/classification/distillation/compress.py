from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import sys
import logging
import paddle
import argparse
import functools
import paddle.fluid as fluid
sys.path.append("..")
import imagenet_reader as reader
import models
sys.path.append("../../")
from utility import add_arguments, print_arguments

from paddle.fluid.contrib.slim import Compressor

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  64*4,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('total_images',     int,  1281167,              "Training image number.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('model',            str,  "MobileNet",          "Set the network to use.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('teacher_model',    str,  None,          "Set the teacher network to use.")
add_arg('teacher_pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('compress_config',  str,  None,                 "The config file for compression with yaml format.")
add_arg('enable_ce',        bool, False,                "If set, run the task with continuous evaluation logs.")

# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def compress(args):
    # add ce
    if args.enable_ce:
        SEED = 1
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # model definition
    model = models.__dict__[args.model]()

    if args.model == 'ResNet34':
        model.prefix_name = 'res34'
        out = model.net(input=image, class_dim=args.class_dim, fc_name='fc_0')
    else:
        out = model.net(input=image, class_dim=args.class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    #print("="*50+"student_model_params"+"="*50)
    #for v in fluid.default_main_program().list_vars():
    #    print(v.name, v.shape)

    val_program = fluid.default_main_program().clone()
    boundaries = [
        args.total_images / args.batch_size * 30, args.total_images /
        args.batch_size * 60, args.total_images / args.batch_size * 90
    ]
    values = [0.1, 0.01, 0.001, 0.0001]
    opt = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values),
        regularization=fluid.regularizer.L2Decay(4e-5))

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    val_reader = paddle.batch(reader.val(), batch_size=args.batch_size)
    val_feed_list = [('image', image.name), ('label', label.name)]
    val_fetch_list = [('acc_top1', acc_top1.name), ('acc_top5', acc_top5.name)]

    train_reader = paddle.batch(
        reader.train(), batch_size=args.batch_size, drop_last=True)
    train_feed_list = [('image', image.name), ('label', label.name)]
    train_fetch_list = [('loss', avg_cost.name)]

    teacher_programs = []
    distiller_optimizer = None

    teacher_model = models.__dict__[args.teacher_model](prefix_name='res50')
    # define teacher program
    teacher_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(teacher_program, startup_program):
        img = teacher_program.global_block()._clone_variable(
            image, force_persistable=False)
        predict = teacher_model.net(img,
                                    class_dim=args.class_dim,
                                    fc_name='fc_0')
    #print("="*50+"teacher_model_params"+"="*50)
    #for v in teacher_program.list_vars():
    #    print(v.name, v.shape)
    #return

    exe.run(startup_program)
    assert args.teacher_pretrained_model and os.path.exists(
        args.teacher_pretrained_model
    ), "teacher_pretrained_model should be set when teacher_model is not None."

    def if_exist(var):
        return os.path.exists(
            os.path.join(args.teacher_pretrained_model, var.name))

    fluid.io.load_vars(
        exe,
        args.teacher_pretrained_model,
        main_program=teacher_program,
        predicate=if_exist)

    distiller_optimizer = opt
    teacher_programs.append(teacher_program.clone(for_test=True))

    com_pass = Compressor(
        place,
        fluid.global_scope(),
        fluid.default_main_program(),
        train_reader=train_reader,
        train_feed_list=train_feed_list,
        train_fetch_list=train_fetch_list,
        eval_program=val_program,
        eval_reader=val_reader,
        eval_feed_list=val_feed_list,
        eval_fetch_list=val_fetch_list,
        teacher_programs=teacher_programs,
        save_eval_model=True,
        prune_infer_model=[[image.name], [out.name]],
        train_optimizer=opt,
        distiller_optimizer=distiller_optimizer)
    com_pass.config(args.compress_config)
    com_pass.run()


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
