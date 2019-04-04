from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import time
import argparse
import functools
import numpy as np
import paddle
import paddle.fluid as fluid
import models
import reader
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('embedding_size', int, 0, "Embedding size.")
add_arg('batch_size', int, 1, "Minibatch size.")
add_arg('image_shape', str, "3,224,224", "Input image size.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('with_mem_opt', bool, False, "Whether to use memory optimization or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def infer(args):
    # parameters from arguments
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')

    # model definition
    model = models.__dict__[model_name]()
    out = model.net(input=image, embedding_size=args.embedding_size)

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

    infer_reader = paddle.batch(reader.infer(args), batch_size=args.batch_size, drop_last=False)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    fetch_list = [out.name]

    for batch_id, data in enumerate(infer_reader()):
        result = exe.run(test_program, fetch_list=fetch_list, feed=feeder.feed(data))
        result = result[0][0].reshape(-1)
        print("Test-{0}-feature: {1}".format(batch_id, result[:5]))
        sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    infer(args)


if __name__ == '__main__':
    main()
