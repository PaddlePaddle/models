#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
from PIL import Image
import paddle.fluid as fluid
import paddle
import numpy as np
from scipy.misc import imsave
import glob
from util.config import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model_net',         str,   'cgan',            "The model used")
add_arg('net_G',             str,   "resnet_9block",   "Choose the CycleGAN generator's network, choose in [resnet_9block|resnet_6block|unet_128|unet_256]")
add_arg('input',             str,   None,              "The images to be infered.")
add_arg('init_model',        str,   None,              "The init model file of directory.")
add_arg('output',            str,   "./infer_result",  "The directory the infer result to be saved to.")
add_arg('input_style',       str,   "A",               "The style of the input, A or B")
add_arg('norm_type',         str,   "batch_norm",      "Which normalization to used")
add_arg('use_gpu',           bool,  True,              "Whether to use GPU to train.")
add_arg('dropout',           bool,  False,             "Whether to use dropout")
add_arg('data_shape',        int,   256,               "The shape of load image")
add_arg('g_base_dims',       int,   64,                "Base channels in CycleGAN generator")
# yapf: enable


def infer(args):
    data_shape = [-1, 3, args.data_shape, args.data_shape]
    input = fluid.layers.data(name='input', shape=data_shape, dtype='float32')
    model_name = 'net_G'
    if args.model_net == 'cyclegan':
        from network.CycleGAN_network import network_G, network_D

        if args.input_style == "A":
            fake = network_G(input, name="GA", cfg=args)
        elif args.input_style == "B":
            fake = network_G(input, name="GB", cfg=args)
        else:
            raise "Input with style [%s] is not supported." % args.input_style
    elif args.model_net == 'cgan':
        pass
    else:
        pass

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    for var in fluid.default_main_program().global_block().all_parameters():
        print(var.name)
    print(args.init_model + '/' + model_name)
    fluid.io.load_persistables(exe, args.init_model + "/" + model_name)
    print('load params done')

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for file in glob.glob(args.input):
        print("read {}".format(file))
        image_name = os.path.basename(file)
        image = Image.open(file).convert('RGB')
        image = image.resize((256, 256), Image.BICUBIC)
        image = np.array(image).transpose([2, 0, 1]).astype('float32')
        image = image / 255.0
        image = (image - 0.5) / 0.5
        data = image[np.newaxis, :]
        tensor = fluid.LoDTensor()
        tensor.set(data, place)

        fake_temp = exe.run(fetch_list=[fake.name], feed={"input": tensor})
        fake_temp = np.squeeze(fake_temp[0]).transpose([1, 2, 0])
        input_temp = np.squeeze(data).transpose([1, 2, 0])

        imsave(args.output + "/fake_" + image_name, (
            (fake_temp + 1) * 127.5).astype(np.uint8))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    infer(args)
