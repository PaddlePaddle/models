import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import models
import argparse
import functools
from losses import tripletloss
from losses import quadrupletloss
from losses import emlloss
from losses.metrics import recall_topk
from utility import add_arguments, print_arguments
import math

# yapf: disable
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',        int,     120,                  "Minibatch size.")
add_arg('use_gpu',           bool,    True,                 "Whether to use GPU or not.")
add_arg('image_shape',       str,     "3,224,224",          "Input image size.")
add_arg('with_mem_opt',      bool,    False,                "Whether to use memory optimization or not.")
add_arg('pretrained_model',  str,     None,                 "Whether to use pretrained model.")
add_arg('model',             str,     "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('loss_name',         str,     "emlloss",            "Loss name.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def eval(args):
    # parameters from arguments
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    loss_name = args.loss_name

    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()
    out = model.net(input=image, class_dim=200)

    if loss_name == "tripletloss":
        metricloss = tripletloss(test_batch_size=args.batch_size, margin=0.1)
        cost = metricloss.loss(out[0])
    elif loss_name == "quadrupletloss":
        metricloss = quadrupletloss(test_batch_size=args.batch_size)
        cost = metricloss.loss(out[0])
    elif loss_name == "emlloss":
        metricloss = emlloss(
            test_batch_size=args.batch_size, samples_each_class=2, fea_dim=2048)
        cost = metricloss.loss(out[0])

    avg_cost = fluid.layers.mean(x=cost)
    test_program = fluid.default_main_program().clone(for_test=True)

    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    test_reader = metricloss.test_reader
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    fetch_list = [avg_cost.name, out[0].name]

    test_info = [[]]
    f = []
    l = []
    for batch_id, (data, label) in enumerate(test_reader()):
        t1 = time.time()
        loss, feas = exe.run(test_program,
                             fetch_list=fetch_list,
                             feed=feeder.feed(data))
        f.append(feas)
        l.append(label)

        t2 = time.time()
        period = t2 - t1
        loss = np.mean(np.array(loss))
        test_info[0].append(loss)
        if batch_id % 10 == 0:
            recall = recall_topk(feas, label, k=1)
            print("testbatch {0}, loss {1}, recall {2}, time {3}".format(  \
                  batch_id, loss, recall, "%2.2f sec" % period))
            sys.stdout.flush()

    test_loss = np.array(test_info[0]).mean()
    f = np.vstack(f)
    l = np.vstack(l)
    recall = recall_topk(f, l, k=1)
    print("End test, test_loss {0}, test recall {1}".format(  \
          test_loss, recall))
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()
