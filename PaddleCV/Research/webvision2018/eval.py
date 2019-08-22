#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from utils import add_arguments, print_arguments, accuracy
import math
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser(description=__doc__)
# yapf: disable
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,  32,                    "Minibatch size.")
add_arg('use_gpu',          bool, True,                 "Whether to use GPU or not.")
add_arg('class_dim',        int,  5000,                 "Class number.")
add_arg('image_shape',      str,  "3,224,224",          "Input image size")
add_arg('pretrained_model', str,  None,                 "Whether to use pretrained model.")
add_arg('model',            str,  "ResNeXt101_32x4d",   "Set the network to use.")
add_arg('img_list',         str,  "None",               "list of valset.")
add_arg('img_path',         str,  "NOne",               "path of valset.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def eval(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    pretrained_model = args.pretrained_model
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

    fetch_list = [out.name]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)


    test_batch_size = args.batch_size

    img_size = image_shape[1]
    test_reader = paddle.batch(reader.test(args, img_size), batch_size=test_batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image])

    targets = []
    with open(args.img_list, 'r') as f:
        for line in f.readlines():
            targets.append(line.strip().split()[-1])
    targets = np.array(targets, dtype=np.int)

    preds = []
    TOPK = 5

    for batch_id, data in enumerate(test_reader()):
        all_result = exe.run(test_program,
                         fetch_list=fetch_list,
                         feed=feeder.feed(data))
        pred_label = np.argsort(-all_result[0], 1)[:,:5]
        print("Test-{0}".format(batch_id))
        preds.append(pred_label)
    preds = np.vstack(preds)
    top1, top5 = accuracy(targets, preds)
    print("top1:{:.4f} top5:{:.4f}".format(top1,top5))

def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()
