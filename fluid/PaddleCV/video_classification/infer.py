import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
from resnet import TSN_ResNet
import reader

import argparse
import functools
from paddle.fluid.framework import Parameter
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_layers',       int,    50,             "How many layers for ResNet model.")
add_arg('with_mem_opt',     bool,   True,           "Whether to use memory optimization or not.")
add_arg('class_dim',        int,    101,            "Number of class.")
add_arg('seg_num',          int,    7,              "Number of segments.")
add_arg('image_shape',      str,    "3,224,224",    "Input image size.")
add_arg('test_model',       str,    None,           "Test model path.")
# yapf: enable


def infer(args):
    # parameters from arguments
    seg_num = args.seg_num
    class_dim = args.class_dim
    num_layers = args.num_layers
    test_model = args.test_model

    if test_model == None:
        print('Please specify the test model ...')
        return

    image_shape = [int(m) for m in args.image_shape.split(",")]
    image_shape = [seg_num] + image_shape

    # model definition
    model = TSN_ResNet(layers=num_layers, seg_num=seg_num)
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')

    out = model.net(input=image, class_dim=class_dim)

    # for test
    inference_program = fluid.default_main_program().clone(for_test=True)

    if args.with_mem_opt:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    def is_parameter(var):
        if isinstance(var, Parameter):
            return isinstance(var, Parameter)

    if test_model is not None:
        vars = filter(is_parameter, inference_program.list_vars())
        fluid.io.load_vars(exe, test_model, vars=vars)

    # reader
    test_reader = paddle.batch(reader.infer(seg_num), batch_size=1)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    fetch_list = [out.name]

    # test
    TOPK = 1
    for batch_id, data in enumerate(test_reader()):
        data, vid = data[0]
        data = [[data]]
        result = exe.run(inference_program,
                         fetch_list=fetch_list,
                         feed=feeder.feed(data))
        result = result[0][0]
        pred_label = np.argsort(result)[::-1][:TOPK]
        print("Test sample: {0}, score: {1}, class {2}".format(vid, result[
            pred_label], pred_label))
        sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    infer(args)


if __name__ == '__main__':
    main()
