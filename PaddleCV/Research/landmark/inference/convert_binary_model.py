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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import functools
import paddle
import paddle.fluid as fluid
import models
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model', str, "ResNet200_vd", "Set the network to use.")
add_arg('embedding_size', int, 512, "Embedding size.")
add_arg('image_shape', str, "3,448,448", "Input image size.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('binary_model', str, None, "Set binary_model dir")
add_arg('task_mode', str, "retrieval", "Set task mode")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def convert(args):
    # parameters from arguments
    model_name = args.model
    pretrained_model = args.pretrained_model
    if not os.path.exists(pretrained_model):
        print("pretrained_model doesn't exist!")
        sys.exit(-1) 
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')

    # model definition
    model = models.__dict__[model_name]()
    if args.task_mode == 'retrieval':
        out = model.net(input=image, embedding_size=args.embedding_size)
    else:
        out = model.net(input=image)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        return os.path.exists(os.path.join(pretrained_model, var.name))
    fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    fluid.io.save_inference_model(
        dirname = args.binary_model,
        feeded_var_names = ['image'],
        target_vars = [out['embedding']] if args.task_mode == 'retrieval' else [out],
        executor = exe,
        main_program = None,
        model_filename = 'model',
        params_filename = 'params')

    print('input_name: {}'.format('image'))
    print('output_name: {}'.format(out['embedding'].name)) if args.task_mode == 'retrieval' else ('output_name: {}'.format(out.name))
    print("convert done.")


def main():
    args = parser.parse_args()
    print_arguments(args)
    convert(args)


if __name__ == '__main__':
    main()
