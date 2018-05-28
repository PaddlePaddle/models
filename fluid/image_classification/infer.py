import os
import sys
import numpy as np
import argparse
import functools

import paddle
import paddle.fluid as fluid
from utility import add_arguments, print_arguments
from se_resnext import SE_ResNeXt
import reader

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   1,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,      "Whether to use GPU or not.")
add_arg('test_list',        str,   '',        "The testing data lists.")
add_arg('num_layers',       int,  50,         "How many layers for SE-ResNeXt model.")
add_arg('model_dir',        str,   '',        "The model path.")
# yapf: enable


def infer(args):
    class_dim = 1000
    image_shape = [3, 224, 224]
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    out = SE_ResNeXt(input=image, class_dim=class_dim, layers=args.num_layers)
    out = fluid.layers.softmax(input=out)

    inference_program = fluid.default_main_program().clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    if not os.path.exists(args.model_dir):
        raise ValueError("The model path [%s] does not exist." %
                         (args.model_dir))
    if not os.path.exists(args.test_list):
        raise ValueError("The test lists [%s] does not exist." %
                         (args.test_list))

    def if_exist(var):
        return os.path.exists(os.path.join(args.model_dir, var.name))

    fluid.io.load_vars(exe, args.model_dir, predicate=if_exist)

    test_reader = paddle.batch(
        reader.infer(args.test_list), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    fetch_list = [out]

    TOPK = 1
    for batch_id, data in enumerate(test_reader()):
        result = exe.run(inference_program,
                         feed=feeder.feed(data),
                         fetch_list=fetch_list)
        result = result[0]
        pred_label = np.argsort(result)[::-1][0][0]
        print("Test {0}-score {1}, class {2}: "
              .format(batch_id, result[0][pred_label], pred_label))
        sys.stdout.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    infer(args)
