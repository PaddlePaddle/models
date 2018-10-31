# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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
from __future__ import unicode_literals

import os
import sys
import shutil
import glob
import numpy as np

sys.path.append("..")
from config.config_parser import ConfigPaser

class WeightParser(object):

    def __init__(self, weight_file, cfg_file, save_dir, conv_num=None):
        self.weight_file = weight_file
        self.cfg_file = cfg_file
        self.save_dir = save_dir
        self.conv_num = conv_num
        self.cfg_parser = ConfigPaser(cfg_file)

    def init_dir(self):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.mkdir(self.save_dir)
        return self.save_dir

    def parse_weight_to_separate_file(self):
        self.save_dir = self.init_dir()
        weights = np.fromfile(
                open(self.weight_file, 'rb'),
                dtype = np.float32)[5:]
        # print("Total weight num: ", weights.shape[0])
        w_idx = 0

        model_defs = self.cfg_parser.parse()
        if model_defs is None:
            return None

        hyperparams = model_defs.pop(0)
        in_channels = [int(hyperparams['channels'])]
        parsed_conv_num = 0
        for i, layer_def in enumerate(model_defs):
            if layer_def['type'] == 'convolutional':
                filters = int(layer_def['filters'])
                size = int(layer_def['size'])
                conv_name = "conv" + str(i)
                if layer_def.get('batch_normalize', 0):
                    bn_name = "bn" + str(i)
                    offset = weights[w_idx: w_idx + filters]
                    offset.tofile(os.path.join(self.save_dir, bn_name+"_offset"))
                    w_idx += filters
                    scale = weights[w_idx: w_idx + filters]
                    scale.tofile(os.path.join(self.save_dir, bn_name+"_scale"))
                    w_idx += filters
                    mean = weights[w_idx: w_idx + filters]
                    mean.tofile(os.path.join(self.save_dir, bn_name+"_mean"))
                    w_idx += filters
                    var = weights[w_idx: w_idx + filters]
                    var.tofile(os.path.join(self.save_dir, bn_name+"_var"))
                    w_idx += filters
                else:
                    conv_bias = weights[w_idx: w_idx + filters]
                    conv_bias.tofile(os.path.join(self.save_dir, conv_name+"_bias"))
                    w_idx += filters
                conv_weight_num = in_channels[-1] * filters * size * size
                conv_weight = weights[w_idx: w_idx + conv_weight_num]
                conv_weight.tofile(os.path.join(self.save_dir, conv_name+"_weights"))
                w_idx += conv_weight_num
                in_channels.append(filters)

                # print(conv_name, "parse weight index: ", w_idx)
                parsed_conv_num += 1
                if self.conv_num is not None:
                    if parsed_conv_num >= self.conv_num:
                        break

            if layer_def['type'] == 'route':
                layers = map(int, layer_def['layers'].split(','))
                out_channel = 0
                for layer in layers:
                    if layer < 0:
                        out_channel += in_channels[layer]
                    else:
                        out_channel += in_channels[layer + 1]
                in_channels.append(out_channel)
            if layer_def['type'] in ['shortcut', 'yolo', 'upsample', 'maxpool']:
                in_channels.append(in_channels[-1])

        assert w_idx == weights.shape[0], "parse imcomplete"

    def convert_file_to_fluid(self):
        filenames = glob.glob(self.save_dir+"/*")
        for filename in filenames:
            src_filename = "./test/" + filename.split("/")[-1]
            assert os.path.exists(src_filename)

            with open(src_filename, 'rb') as f:
                src_data = f.read()
            with open(filename, 'rb') as f:
                data = f.read()
            head_len = len(src_data) - len(data)
            with open(filename, 'wb') as f:
                f.write(src_data[:head_len])
                f.write(data)

    def check_conver_result(self):
        filenames = glob.glob(self.save_dir+"/*")
        for filename in filenames:
            src_filename = "./test/" + filename.split("/")[-1]
            assert os.path.exists(src_filename)

            f = np.fromfile(open(filename, 'rb'), dtype=np.int8)
            sf = np.fromfile(open(src_filename, 'rb'), dtype=np.int8)

            assert f.shape == sf.shape, "check {} failed {}, {}".format(filename, f.shape, sf.shape)


if __name__ == "__main__":
    model = sys.argv[1]
    if model == "pretrain":
        weight_path = "darknet53.pretrain"
        cfg_path = "../config/yolov3.cfg"
        conv_num = 53 - 1
    else:
        weight_path = model + '.weights'
        cfg_path = "../config/" + model + ".cfg"
        conv_num = None
    for path in [weight_path, cfg_path]:
        if not os.path.isfile(path):
            print(path, "not found!")
            exit()
    wp = WeightParser(weight_path, cfg_path, model, conv_num)
    wp.parse_weight_to_separate_file()
    wp.convert_file_to_fluid()
    wp.check_conver_result()

