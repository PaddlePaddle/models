#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys
import time
import shutil
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

from models import *
from data.data_utils import *
from data.modelnet40_reader import ModelNet40ClsReader 
from utils import *

logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

np.random.seed(1024)


def parse_args():
    parser = argparse.ArgumentParser("PointNet++ semantic segmentation train script")
    parser.add_argument(
        '--model',
        type=str,
        default='MSG',
        help='SSG or MSG model to train, default MSG')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='evaluation batch size, default 1')
    parser.add_argument(
        '--num_points',
        type=int,
        default=2048,
        help='number of points in a sample, default: 4096')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=40,
        help='number of classes in dataset, default: 13')
    parser.add_argument(
        '--weights',
        type=str,
        default='checkpoints/200',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset/ModelNet40/modelnet40_ply_hdf5_2048',
        help='dataset directory')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='mini-batch interval for logging.')
    args = parser.parse_args()
    return args


def eval():
    args = parse_args()
    print_arguments(args)
    # check whether the installed paddle is compiled with GPU
    check_gpu(args.use_gpu)

    assert args.model in ['MSG', 'SSG'], \
            "--model can only be 'MSG' or 'SSG'"

    # build model
    startup = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup):
        with fluid.unique_name.guard():
            eval_model = PointNet2ClsMSG(args.num_classes, args.num_points) \
                           if args.model == 'MSG' else \
                         PointNet2ClsSSG(args.num_classes, args.num_points)
            eval_model.build_model()
            eval_feeds = eval_model.get_feeds()
            eval_outputs = eval_model.get_outputs()
            eval_loader = eval_model.get_loader()
    eval_prog = eval_prog.clone(True)
    eval_keys, eval_values = parse_outputs(eval_outputs)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup)

    if not os.path.isdir(args.weights):
        assert os.path.exists("{}.pdparams".format(args.weights)), \
                "Given resume weight {}.pdparams not exist.".format(args.weights)
    fluid.load(eval_prog, args.weights, exe)

    eval_compile_prog = fluid.compiler.CompiledProgram(eval_prog)
    
    # get reader
    modelnet_reader = ModelNet40ClsReader(args.data_dir, mode='test')
    eval_reader = modelnet_reader.get_reader(args.batch_size, args.num_points)
    eval_loader.set_sample_list_generator(eval_reader, place)

    eval_stat = Stat()
    try:
        eval_loader.start()
        eval_iter = 0
        eval_periods = []
        while True:
            cur_time = time.time()
            eval_outs = exe.run(eval_compile_prog, fetch_list=eval_values)
            period = time.time() - cur_time
            eval_periods.append(period)
            eval_stat.update(eval_keys, eval_outs)
            if eval_iter % args.log_interval == 0:
                log_str = ""
                for name, value in zip(eval_keys, eval_outs):
                    log_str += "{}: {:.4f}, ".format(name, np.mean(value))
                logger.info("[EVAL] batch {}: {}time: {:.2f}".format(eval_iter, log_str, period))
            eval_iter += 1
    except fluid.core.EOFException:
        logger.info("[EVAL] Eval finished, {}average time: {:.2f}".format(eval_stat.get_mean_log(), np.mean(eval_periods[1:])))
    finally:
        eval_loader.reset()


if __name__ == "__main__":
    eval()
