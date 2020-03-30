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

from args import print_arguments, parse_args
from utils import tdm_sampler_prepare, tdm_child_prepare
from train_network import TdmTrainNet

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def get_dataset(inputs, args):
    """
    get dataset
    """
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python ./dataset_generator.py")
    dataset.set_batch_size(args.batch_size)
    dataset.set_thread(int(args.cpu_num))
    file_list = [
        str(args.train_files_path) + "/%s" % x
        for x in os.listdir(args.train_files_path)
    ]
    dataset.set_filelist(file_list)
    logger.info("file list: {}".format(file_list))
    return dataset


def run_train(args):
    """
    run train
    """
    logger.info("TDM Begin build network.")
    tdm_model = TdmTrainNet(args)
    inputs = tdm_model.input_data()
    avg_cost, acc = tdm_model.tdm(inputs)
    logger.info("TDM End build network.")

    dataset = get_dataset(inputs, args)

    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=args.learning_rate,
        lazy_mode=True)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # Set TDM_Tree Parameter
    Numpy_model = {}
    Numpy_model['TDM_Tree_Travel'] = tdm_model.travel_array
    Numpy_model['TDM_Tree_Layer'] = tdm_model.layer_array
    Numpy_model['TDM_Tree_Info'] = tdm_model.info_array
    for param_name in Numpy_model:
        param_t = fluid.global_scope().find_var(param_name).get_tensor()
        param_t.set(Numpy_model[str(param_name)].astype('int32'), place)

    if args.load_model:
        path = args.init_model_files_path
        fluid.io.load_persistables(
            executor=exe,
            dirname=path,
            main_program=fluid.default_main_program())
        lr = fluid.global_scope().find_var("learning_rate_0").get_tensor()
        lr.set(np.array(args.learning_rate).astype('float32'), place)
        logger.info("Load persistables from \"{}\"".format(path))

    if args.save_init_model:
        logger.info("Begin Save Init model.")
        model_path = os.path.join(args.model_files_path, "init_model")
        fluid.io.save_persistables(executor=exe, dirname=model_path)
        logger.info("End Save Init model.")

    logger.info("TDM Local training begin ...")
    for epoch in range(args.epoch_num):
        start_time = time.time()
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=dataset,
            fetch_list=[acc, avg_cost],
            fetch_info=["Epoch {} acc".format(
                epoch), "Epoch {} loss".format(epoch)],
            print_period=10,
            debug=False,
        )
        end_time = time.time()
        logger.info("Epoch %d finished, use time=%d sec\n" %
                    (epoch, end_time - start_time))

        model_path = os.path.join(args.model_files_path, "epoch_" + str(epoch))
        fluid.io.save_persistables(executor=exe, dirname=model_path)

    logger.info("Local training success!")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    run_train(args)
