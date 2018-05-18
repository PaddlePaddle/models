import numpy as np
import argparse
import functools

import paddle
import paddle.fluid as fluid
from pyramidbox import PyramidBox
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   32,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('parallel',         bool,  True,      "Parallel.")
#yapf: enable

def train(args,
          learning_rate,
          batch_size):

    network = PyramidBox([3, 640, 640])
    face_loss, head_loss = network.train()
    loss = face_loss + head_loss

    test_program, face_map_eval, head_map_eval = network.test()

    epocs = 19200 / batch_size
    boundaries = [epocs * 40, epocs * 60, epocs * 80, epocs * 100]
    lr = learning_rate
    values = [lr, lr * 0.5, lr * 0.25, lr * 0.1, lr * 0.01]
    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    optimizer.minimize(loss)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    #print(fluid.default_main_program())
    #print(test_program)


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    train(args,
          learning_rate=0.01,
          batch_size=args.batch_size)
