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
add_arg('batch_size',       int,  64*4,                "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('model',            str,  None,                "The target model")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('config_file',      str,  None,                "The config file for compression with yaml format.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def compress(args):
    image_shape = "3,224,224"
    image_shape = [int(m) for m in image_shape.split(",")]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # model definition
    model = models.__dict__[args.model]()

    out = model.net(input=image, class_dim=1000)
    # print(out)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    val_program = fluid.default_main_program().clone()

    # quantization usually use small learning rate
    values = [1e-4, 1e-5, 1e-6]
    opt = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[5000 * 30, 5000 * 60], values=values),
        regularization=fluid.regularizer.L2Decay(4e-5))

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:
        assert os.path.exists(
            args.pretrained_model), "pretrained_model path doesn't exist"

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
        teacher_programs=[],
        train_optimizer=opt,
        distiller_optimizer=None)
    com_pass.config(args.config_file)
    com_pass.run()


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
