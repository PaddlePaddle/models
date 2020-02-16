import os
import sys
import logging
import paddle
import argparse
import functools
import math
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
add_arg('model',            str,  None,                "The target model.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('lr',               float,  0.1,               "The learning rate used to fine-tune pruned model.")
add_arg('lr_strategy',      str,  "piecewise_decay",   "The learning rate decay strategy.")
add_arg('l2_decay',         float,  3e-5,               "The l2_decay parameter.")
add_arg('momentum_rate',    float,  0.9,               "The value of momentum_rate.")
add_arg('num_epochs',       int,  120,               "The number of total epochs.")
add_arg('total_images',     int,  1281167,               "The number of total training images.")
parser.add_argument('--step_epochs', nargs='+', type=int, default=[30, 60, 90], help="piecewise decay step")
add_arg('config_file',      str, None,                 "The config file for compression with yaml format.")
add_arg('enable_ce', bool, False, "If set, run the task with continuous evaluation logs.")
# yapf: enable


model_list = [m for m in dir(models) if "__" not in m]

def piecewise_decay(args):
    step = int(math.ceil(float(args.total_images) / args.batch_size))
    bd = [step * e for e in args.step_epochs]
    lr = [args.lr * (0.1**i) for i in range(len(bd) + 1)]
    learning_rate = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        regularization=fluid.regularizer.L2Decay(args.l2_decay))
    return optimizer

def cosine_decay(args):
    step = int(math.ceil(float(args.total_images) / args.batch_size))
    learning_rate = fluid.layers.cosine_decay(
        learning_rate=args.lr,
        step_each_epoch=step,
        epochs=args.num_epochs)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=args.momentum_rate,
        regularization=fluid.regularizer.L2Decay(args.l2_decay))
    return optimizer

def create_optimizer(args):
    if args.lr_strategy == "piecewise_decay":
        return piecewise_decay(args)
    elif args.lr_strategy == "cosine_decay":
        return cosine_decay(args)

def compress(args):
    # add ce
    if args.enable_ce:
        SEED = 1
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

    class_dim=1000
    image_shape="3,224,224"
    image_shape = [int(m) for m in image_shape.split(",")]
    assert args.model in model_list, "{} is not in lists: {}".format(args.model, model_list)
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # model definition
    model = models.__dict__[args.model]()
    out = model.net(input=image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    val_program = fluid.default_main_program().clone()
    opt = create_optimizer(args)
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
        save_eval_model=True,
        prune_infer_model=[[image.name], [out.name]],
        train_optimizer=opt)
    com_pass.config(args.config_file)
    com_pass.run()


def main():
    args = parser.parse_args()
    print_arguments(args)
    compress(args)


if __name__ == '__main__':
    main()
