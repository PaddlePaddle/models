import os
import numpy as np
import time
import sys
import paddle.v2 as paddle
import paddle.fluid as fluid
from resnet import ResNet 
import reader

import argparse
import functools
from paddle.fluid.framework import Parameter
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,    128,            "Minibatch size.")
add_arg('num_layers',       int,    50,             "How many layers for ResNet model.")
add_arg('with_mem_opt',     bool,   True,           "Whether to use memory optimization or not.")
add_arg('num_epochs',       int,    60,             "Number of epochs.")
add_arg('class_dim',        int,    101,            "Number of class.")
add_arg('seg_num',          int,    7,              "Number of segments.")
add_arg('image_shape',      str,    "3,224,224",    "Input image size.")
add_arg('model_save_dir',   str,    "output",       "Model save directory.")
add_arg('pretrained_model', str,    None,           "Whether to use pretrained model.")
add_arg('test_model',       str,    None,           "Test model path.")
add_arg('total_videos',     int,    9537,           "Training video number.")
add_arg('lr_init',          float,  0.01,           "Set initial learning rate.")
# yapf: enable

def infer(args):
    # parameters from arguments
    seg_num = args.seg_num
    class_dim = args.class_dim
    num_layers = args.num_layers
    batch_size = args.batch_size
    test_model = args.test_model

    if test_model == None:
        print ('Please specify the test model ...')
        return

    image_shape = [int(m) for m in args.image_shape.split(",")]
    image_shape = [seg_num] + image_shape

    # model definition
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')

    out = ResNet(input=image, seg_num=seg_num, class_dim=class_dim, layers=num_layers)
    
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
        result = exe.run(inference_program, fetch_list=fetch_list, feed=feeder.feed(data))
        result = result[0][0]
        pred_label = np.argsort(result)[::-1][:TOPK]
        print ("Test sample: {0}, score: {1}, class {2}".format(vid, result[pred_label], pred_label))
        sys.stdout.flush()

def main():
    args = parser.parse_args()
    print_arguments(args)
    infer(args)

if __name__ == '__main__':
    main()
