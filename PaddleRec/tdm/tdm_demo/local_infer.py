# -*- coding=utf-8 -*-
"""
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function
import os
import time
import numpy as np
import logging
import argparse

import paddle
import paddle.fluid as fluid
from paddle.fluid import profiler

from args import print_arguments, parse_args
from infer_network import TdmInferNet
from dataset_generator import TDMDataset

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def to_tensor(data, place):
    """
    Convert data to paddle tensor
    """
    flattened_data = np.concatenate(data, axis=0).astype("float32")
    flattened_data = flattened_data.reshape([-1, 768])
    res = fluid.Tensor()
    res.set(flattened_data, place)
    return res


def data2tensor(data, place):
    """
    Dataset prepare
    """
    input_emb = to_tensor([x[0] for x in data], place)

    return input_emb


def run_infer(args, model_path):
    """run infer"""
    logger.info("Infer Begin")
    file_list = [
        str(args.test_files_path) + "/%s" % x
        for x in os.listdir(args.test_files_path)
    ]

    tdm_model = TdmInferNet(args)
    inputs = tdm_model.input_data()
    res_item = tdm_model.infer_net(inputs)
    test_reader = TDMDataset().infer_reader(file_list, args.batch_size)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    path = os.path.join(args.model_files_path, model_path)
    fluid.io.load_persistables(
        executor=exe,
        dirname=path,
        main_program=fluid.default_main_program())

    logger.info("Load persistables from \"{}\"".format(path))

    if args.save_init_model:
        logger.info("Begin Save infer model.")
        model_path = (str(args.model_files_path) + "/" + "infer_model")
        fluid.io.save_inference_model(executor=exe, dirname=model_path,
                                      feeded_var_names=[
                                          'input_emb', 'first_layer_node', 'first_layer_node_mask'],
                                      target_vars=[res_item])
        logger.info("End Save infer model.")

    first_layer_node = tdm_model.first_layer_node
    first_layer_nums = len(first_layer_node)
    first_layer_node = np.array(first_layer_node)
    first_layer_node = first_layer_node.reshape((1, -1)).astype('int64')
    first_layer_node = first_layer_node.repeat(args.batch_size, axis=0)
    # 在demo中，假设infer起始层的节点都不是叶子节点，mask=0
    # 若真实的起始层含有叶子节点，则对应位置的 mask=1
    first_layer_mask = (
        np.zeros((args.batch_size, first_layer_nums))).astype('int64')

    for batch_id, data in enumerate(test_reader()):
        input_emb = data2tensor(data, place)
        item_res = exe.run(fluid.default_main_program(),
                           feed={"input_emb": input_emb,
                                 "first_layer_node": first_layer_node,
                                 "first_layer_node_mask": first_layer_mask},
                           fetch_list=[res_item])
        logger.info("TEST --> batch: {} infer_item {}".format(
            batch_id, item_res))
    logger.info("Inference complete!")


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    args = parse_args()
    print_arguments(args)
    # 在此处指定infer模型所在的文件夹
    path = "epoch_0"
    run_infer(args, path)
