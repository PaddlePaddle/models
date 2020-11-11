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
import argparse
import functools
import paddle
import paddle.fluid as fluid
import models
from utility import add_arguments, print_arguments, check_cuda

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('embedding_size', int, 0, "Embedding size.")
add_arg('image_shape', str, "3,224,224", "Input image size.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('model_save_dir', str, 'save_inference_model', "Whether to save the inference model.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def save_inference_model(args):
    # parameters from arguments
    model_name = args.model
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.data(name='image', shape=[None] + image_shape, dtype='float32')

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

        fluid.load(model_path=pretrained_model, program=test_program, executor=exe)

        print('Saving the inference model...')
        
        fluid.io.save_inference_model(
            dirname=model_save_dir, 
            feeded_var_names=['image'], 
            target_vars=[out], 
            executor=exe, 
            params_filename='__params__'
        )
        print('Finish.')

    else:
        print('Can\'t load the pretrained_model. Please set the true pretrained_model dir.')

def main():
    paddle.enable_static()
    args = parser.parse_args()
    print_arguments(args)
    check_cuda(args.use_gpu)
    save_inference_model(args)

if __name__ == '__main__':
    main()
