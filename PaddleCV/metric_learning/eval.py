#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from utility import add_arguments, print_arguments, check_cuda
from utility import fmt_time, recall_topk

# yapf: disable
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('embedding_size', int, 0, "Embedding size.")
add_arg('batch_size', int, 10, "Minibatch size.")
add_arg('image_shape', str, "3,224,224", "Input image size.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def eval(args):
    # parameters from arguments
    model_name = args.model
    pretrained_model = args.pretrained_model
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.data(name='image', shape=[None] + image_shape, dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')

    test_loader = fluid.io.DataLoader.from_generator(
                feed_list=[image, label],
                capacity=64,
                use_double_buffer=True,
                iterable=True)

    # model definition
    model = models.__dict__[model_name]()
    out = model.net(input=image, embedding_size=args.embedding_size)

    test_program = fluid.default_main_program().clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.load(program=test_program, model_path=pretrained_model, executor=exe)

    test_loader.set_sample_generator(
        reader.test(args),
        batch_size=args.batch_size,
        drop_last=False,
        places=place)

    fetch_list = [out.name]

    f, l = [], []
    for batch_id, data in enumerate(test_loader()):
        t1 = time.time()
        [feas] = exe.run(test_program, fetch_list=fetch_list, feed=data)
        label = np.asarray(data[0]['label'])
        label = np.squeeze(label)
 
        f.append(feas)
        l.append(label)

        t2 = time.time()
        period = t2 - t1
        if batch_id % 20 == 0:
            print("[%s] testbatch %d, time %2.2f sec" % \
                    (fmt_time(), batch_id, period))

    f = np.vstack(f)
    l = np.hstack(l)
    recall = recall_topk(f, l, k=1)
    print("[%s] End test %d, test_recall %.5f" % (fmt_time(), len(f), recall))
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_cuda(args.use_gpu)
    eval(args)


if __name__ == '__main__':
    main()
