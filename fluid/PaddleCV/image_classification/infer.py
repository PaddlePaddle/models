from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import models
import reader
import argparse
import functools
from models.learning_rate import cosine_decay
from utility import add_arguments, print_arguments
import math

parser = argparse.ArgumentParser(description=__doc__)
# yapf: disable
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_gpu',          bool, True,                 "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                 "Class number.")
add_arg('image_shape',      str,  "3,224,224",          "Input image size")
add_arg('with_mem_opt',     bool, True,                 "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,  None,                 "Whether to use pretrained model.")
add_arg('model',            str,  "SE_ResNeXt50_32x4d", "Set the network to use.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def infer(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')

    # model definition
    model = models.__dict__[model_name]()

    if model_name is "GoogleNet":
        out, _, _ = model.net(input=image, class_dim=class_dim)
    else:
        out = model.net(input=image, class_dim=class_dim)

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

    test_batch_size = 1
    test_reader = paddle.batch(reader.test(), batch_size=test_batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    fetch_list = [out.name]

    TOPK = 1
    for batch_id, data in enumerate(test_reader()):
        result = exe.run(test_program,
                         fetch_list=fetch_list,
                         feed=feeder.feed(data))
        result = result[0][0]
        pred_label = np.argsort(result)[::-1][:TOPK]
        print("Test-{0}-score: {1}, class {2}"
              .format(batch_id, result[pred_label], pred_label))
        sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    infer(args)


if __name__ == '__main__':
    main()
