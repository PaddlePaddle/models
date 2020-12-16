# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import yaml
import os
import io
import random
import time
import numpy as np
from easydict import EasyDict as edict

import paddle
from pgl.contrib.imperative.graph_tensor import GraphTensor

from models import ErnieSageForLinkPrediction
from data import GraphDataset, TrainData, PredictData, GraphDataLoader
from paddlenlp.utils.log import logger


def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    paddle.seed(config.seed)


def do_train(config):
    paddle.set_device("gpu" if config.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(config)

    graphs = [GraphTensor() for x in range(len(config.samples))]
    mode = 'train'
    data = TrainData(config.graph_work_path)
    model = ErnieSageForLinkPrediction.from_pretrained(
        config.model_name_or_path, config=config)
    model = paddle.DataParallel(model)

    train_dataset = GraphDataset(graphs, data, config.batch_size,
                                 config.samples, mode, config.graph_work_path)
    graph_loader = GraphDataLoader(train_dataset)

    optimizer = paddle.optimizer.Adam(
        learning_rate=config.lr, parameters=model.parameters())

    global_step = 0
    tic_train = time.time()
    for epoch in range(config.epoch):
        for step, (graphs, datas) in enumerate(graph_loader()):
            global_step += 1
            loss, outputs = model(graphs, datas)
            if global_step % config.log_per_step == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       config.log_per_step / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()
            if global_step % config.save_per_step == 0:
                if (not config.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(config.output_path,
                                              "model_%d" % global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model._layers.save_pretrained(output_dir)


def tostr(data_array):
    return " ".join(["%.5lf" % d for d in data_array])


def do_predict(config):
    paddle.set_device("gpu" if config.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(config)

    graphs = [GraphTensor() for x in range(len(config.samples))]
    mode = 'predict'
    num_nodes = int(
        np.load(os.path.join(config.graph_work_path, "num_nodes.npy")))
    data = PredictData(num_nodes)
    model = ErnieSageForLinkPrediction.from_pretrained(
        config.model_name_or_path, config=config)
    model = paddle.DataParallel(model)

    train_dataset = GraphDataset(
        graphs,
        data,
        config.batch_size,
        config.samples,
        mode,
        config.graph_work_path,
        shuffle=False)
    graph_loader = GraphDataLoader(train_dataset)

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    id2str = io.open(
        os.path.join(config.graph_work_path, "terms.txt"),
        encoding=config.encoding).readlines()
    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)
    fout = io.open(
        "%s/part-%s" % (config.output_path, trainer_id), "w", encoding="utf8")

    global_step = 0
    epoch = 0
    tic_train = time.time()
    for step, (graphs, datas) in enumerate(graph_loader()):
        global_step += 1
        loss, outputs = model(graphs, datas)
        for user_feat, user_real_index in zip(outputs[0].numpy(),
                                              outputs[3].numpy()):
            # user_feat, user_real_index = 
            sri = id2str[int(user_real_index)].strip("\n")
            line = "{}\t{}\n".format(sri, tostr(user_feat))
            fout.write(line)
        if global_step % config.log_per_step == 0:
            logger.info(
                "predict step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                % (global_step, epoch, step, loss,
                   config.log_per_step / (time.time() - tic_train)))
            tic_train = time.time()
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    parser.add_argument("--do_predict", action='store_true', default=False)
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    logger.info(config)
    if args.do_predict:
        do_func = do_predict
    else:
        do_func = do_train

    if config.n_gpu > 1 and paddle.fluid.core.is_compiled_with_cuda(
    ) and paddle.fluid.core.get_cuda_device_count() > 1:
        paddle.distributed.spawn(do_func, args=(config, ), nprocs=config.n_gpu)
    else:
        do_func(config)
