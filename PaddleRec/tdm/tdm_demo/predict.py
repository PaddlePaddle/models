# -*- coding: utf-8 -*-
"""
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
"""

import os
import argparse
import numpy as np
import argparse
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
from dataset_generator import TDMDataset
from infer_network import TdmInferNet
from args import print_arguments, parse_args
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def to_tensor(data):
    """
    Convert data to paddle tensor
    """
    flattened_data = np.concatenate(data, axis=0).astype("float32")
    flattened_data = flattened_data.reshape([-1, 768])
    return flattened_data


def data2tensor(data):
    """
    Dataset prepare
    """
    input_emb = to_tensor([x[0] for x in data])

    return input_emb


def tdm_input(input_emb, first_layer_node, first_layer_mask):
    """
    Create input of tdm pred
    """
    input_emb = PaddleTensor(input_emb)
    first_layer_node = PaddleTensor(first_layer_node)
    first_layer_mask = PaddleTensor(first_layer_mask)
    return [input_emb, first_layer_node, first_layer_mask]


def main():
    """Predictor main"""
    args = parse_args()

    config = AnalysisConfig(args.model_files_path)
    config.disable_gpu()
    config.enable_profile()
    # config.enable_mkldnn()
    config.set_cpu_math_library_num_threads(args.cpu_num)

    predictor = create_paddle_predictor(config)
    tdm_model = TdmInferNet(args)
    first_layer_node = tdm_model.first_layer_node
    first_layer_nums = len(first_layer_node)
    first_layer_node = np.array(first_layer_node)
    first_layer_node = first_layer_node.reshape((1, -1)).astype('int64')
    first_layer_node = first_layer_node.repeat(args.batch_size, axis=0)
    first_layer_mask = (
        np.zeros((args.batch_size, first_layer_nums))).astype('int64')

    file_list = [
        str(args.test_files_path) + "/%s" % x
        for x in os.listdir(args.test_files_path)
    ]
    test_reader = TDMDataset().infer_reader(file_list, args.batch_size)

    for batch_id, data in enumerate(test_reader()):
        input_emb = data2tensor(data)

        inputs = tdm_input(input_emb, first_layer_node, first_layer_mask)
        outputs = predictor.run(inputs)
        output = outputs[0]
        output_data = output.as_ndarray()

        logger.info("TEST --> batch: {} infer_item {}".format(
            batch_id, output_data))


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    main()
