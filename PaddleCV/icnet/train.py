"""Trainer for ICNet model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from icnet import icnet
import cityscape
import argparse
import functools
import sys
import os
import time
import paddle.fluid as fluid
import numpy as np
from utils import add_arguments, print_arguments, get_feeder_data
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu

if 'ce_mode' in os.environ:
    np.random.seed(10)
    fluid.default_startup_program().random_seed = 90

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   16,         "Minibatch size.")
add_arg('checkpoint_path',   str,   None,       "Checkpoint svae path.")
add_arg('init_model',        str,   None,       "Pretrain model path.")
add_arg('use_gpu',           bool,  True,       "Whether use GPU to train.")
add_arg('random_mirror',     bool,  True,       "Whether prepare by random mirror.")
add_arg('random_scaling',    bool,  True,       "Whether prepare by random scaling.")
# yapf: enable

LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0
LEARNING_RATE = 0.003
POWER = 0.9
LOG_PERIOD = 100
CHECKPOINT_PERIOD = 100
TOTAL_STEP = 100

no_grad_set = []


def create_loss(predict, label, mask, num_classes):
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    label = fluid.layers.reshape(label, shape=[-1, 1])
    predict = fluid.layers.gather(predict, mask)
    label = fluid.layers.gather(label, mask)
    label = fluid.layers.cast(label, dtype="int64")
    loss = fluid.layers.softmax_with_cross_entropy(predict, label)
    no_grad_set.append(label.name)
    return fluid.layers.reduce_mean(loss)


def poly_decay():
    global_step = _decay_step_counter()
    with init_on_cpu():
        decayed_lr = LEARNING_RATE * (fluid.layers.pow(
            (1 - global_step / TOTAL_STEP), POWER))
    return decayed_lr


def train(args):
    data_shape = cityscape.train_data_shape()
    num_classes = cityscape.num_classes()
    # define network
    images = fluid.layers.data(name='image', shape=data_shape, dtype='float32')
    label_sub1 = fluid.layers.data(name='label_sub1', shape=[1], dtype='int32')
    label_sub2 = fluid.layers.data(name='label_sub2', shape=[1], dtype='int32')
    label_sub4 = fluid.layers.data(name='label_sub4', shape=[1], dtype='int32')
    mask_sub1 = fluid.layers.data(name='mask_sub1', shape=[-1], dtype='int32')
    mask_sub2 = fluid.layers.data(name='mask_sub2', shape=[-1], dtype='int32')
    mask_sub4 = fluid.layers.data(name='mask_sub4', shape=[-1], dtype='int32')

    sub4_out, sub24_out, sub124_out = icnet(
        images, num_classes, np.array(data_shape[1:]).astype("float32"))
    loss_sub4 = create_loss(sub4_out, label_sub4, mask_sub4, num_classes)
    loss_sub24 = create_loss(sub24_out, label_sub2, mask_sub2, num_classes)
    loss_sub124 = create_loss(sub124_out, label_sub1, mask_sub1, num_classes)
    reduced_loss = LAMBDA1 * loss_sub4 + LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124

    regularizer = fluid.regularizer.L2Decay(0.0001)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=poly_decay(), momentum=0.9, regularization=regularizer)
    _, params_grads = optimizer.minimize(reduced_loss, no_grad_set=no_grad_set)

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    if args.init_model is not None:
        print("load model from: %s" % args.init_model)

        def if_exist(var):
            return os.path.exists(os.path.join(args.init_model, var.name))

        fluid.io.load_vars(exe, args.init_model, predicate=if_exist)

    iter_id = 0
    t_loss = 0.
    sub4_loss = 0.
    sub24_loss = 0.
    sub124_loss = 0.
    train_reader = cityscape.train(
        args.batch_size, flip=args.random_mirror, scaling=args.random_scaling)
    start_time = time.time()
    while True:
        # train a pass
        for data in train_reader():
            if iter_id > TOTAL_STEP:
                end_time = time.time()
                print("kpis	train_duration	%f" % (end_time - start_time))
                return
            iter_id += 1
            results = exe.run(
                feed=get_feeder_data(data, place),
                fetch_list=[reduced_loss, loss_sub4, loss_sub24, loss_sub124])
            t_loss += results[0]
            sub4_loss += results[1]
            sub24_loss += results[2]
            sub124_loss += results[3]
            # training log
            if iter_id % LOG_PERIOD == 0:
                print(
                    "Iter[%d]; train loss: %.3f; sub4_loss: %.3f; sub24_loss: %.3f; sub124_loss: %.3f"
                    % (iter_id, t_loss / LOG_PERIOD, sub4_loss / LOG_PERIOD,
                       sub24_loss / LOG_PERIOD, sub124_loss / LOG_PERIOD))
                print("kpis	train_cost	%f" % (t_loss / LOG_PERIOD))

                t_loss = 0.
                sub4_loss = 0.
                sub24_loss = 0.
                sub124_loss = 0.
                sys.stdout.flush()

            if iter_id % CHECKPOINT_PERIOD == 0 and args.checkpoint_path is not None:
                dir_name = args.checkpoint_path + "/" + str(iter_id)
                fluid.io.save_persistables(exe, dirname=dir_name)
                print("Saved checkpoint: %s" % (dir_name))


def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args)


if __name__ == "__main__":
    main()
