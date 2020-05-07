# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import random
import sys
import paddle
import argparse
import functools
import time
import numpy as np
import glob
from PIL import Image
from scipy.misc import imsave
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import data_reader
from utility import add_arguments, print_arguments, ImagePool
from trainer import *
from paddle.fluid.dygraph.base import to_variable
import six
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)


# yapf: disable
add_arg('input',             str,   "./image/testA/123_A.jpg",      "input image")
add_arg('output',            str,   "./output_0", "The directory the model and the test result to be saved to.")
add_arg('init_model',        str,   './output_0/checkpoints/0',       "The init model file of directory.")
add_arg('input_style',       str,   "A",        "A or B")
def infer():
    with fluid.dygraph.guard():
        data_shape = [-1,3,256,256]
       
        out_path = args.output + "/single"
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cycle_gan = Cycle_Gan(3)
        save_dir = args.init_model 
        restore, _ = fluid.load_dygraph(save_dir)
        cycle_gan.set_dict(restore)
        cycle_gan.eval()
        for file in glob.glob(args.input):
            print ("read %s" % file)
            image_name = os.path.basename(file)
            image = Image.open(file).convert('RGB')
            image = image.resize((256, 256), Image.BICUBIC)
            image = np.array(image) / 127.5 - 1

            image = image[:, :, 0:3].astype("float32")
            data = image.transpose([2, 0, 1])[np.newaxis,:]

            
            data_A_tmp = to_variable(data)

            fake_A_temp,fake_B_temp,cyc_A_temp,cyc_B_temp,g_A_loss,g_B_loss,idt_loss_A,idt_loss_B,cyc_A_loss,cyc_B_loss,g_loss = cycle_gan(data_A_tmp,data_A_tmp,True,False,False)
       
            fake_A_temp = np.squeeze(fake_A_temp.numpy()[0]).transpose([1, 2, 0])
            fake_B_temp = np.squeeze(fake_B_temp.numpy()[0]).transpose([1, 2, 0])

            if args.input_style == "A":
                imsave(out_path + "/fakeB_" + image_name, (
                    (fake_B_temp + 1) * 127.5).astype(np.uint8))
            if args.input_style == "B":
                imsave(out_path + "/fakeA_" + image_name, (
                    (fake_A_temp + 1) * 127.5).astype(np.uint8))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    infer()
