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
import argparse
import functools
import os
from PIL import Image
import paddle.fluid as fluid
import paddle
import numpy as np
from scipy.misc import imsave
from model import build_generator_resnet_9blocks, build_gen_discriminator
import glob
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('input',             str,   None, "The images to be infered.")
add_arg('output',            str,   "./infer_result", "The directory the infer result to be saved to.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('input_style',        str,  "A",       "The style of the input, A or B")
add_arg('use_gpu',           bool,  True,       "Whether to use GPU to train.")
# yapf: enable


def infer(args):
    data_shape = [-1, 3, 256, 256]
    input = fluid.layers.data(name='input', shape=data_shape, dtype='float32')
    if args.input_style == "A":
        model_name = 'g_a'
        fake = build_generator_resnet_9blocks(input, name="g_A")
    elif args.input_style == "B":
        model_name = 'g_b'
        fake = build_generator_resnet_9blocks(input, name="g_B")
    else:
        raise "Input with style [%s] is not supported." % args.input_style
    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.io.load_persistables(exe, args.init_model + "/" + model_name)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for file in glob.glob(args.input):
        image_name = os.path.basename(file)
        image = Image.open(file)
        image = image.resize((256, 256))
        image = np.array(image) / 127.5 - 1
        if len(image.shape) != 3:
            continue
        data = image.transpose([2, 0, 1])[np.newaxis, :].astype("float32")
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
