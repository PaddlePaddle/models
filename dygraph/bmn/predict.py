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

import paddle
from paddle.io import DataLoader, DistributedBatchSampler
import numpy as np
import argparse
import sys
import os
import ast
import json

from model import BMN
from eval import gen_props
from reader import BmnDataset
from bmn_utils import bmn_post_processing
from config_utils import *

DATATYPE = 'float32'

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("BMN test for performance evaluation.")
    parser.add_argument(
        '--config_file',
        type=str,
        default='bmn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default="checkpoint/bmn_paddle_dy_final",
        help='weight path, None to automatically download weights provided by Paddle.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default="predict_results/",
        help='output dir path, default to use ./predict_results/')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def get_dataset_dict(cfg):
    file_list = cfg.INFER.filelist
    annos = json.load(open(file_list))
    video_dict = {}
    for video_name in annos.keys():
        video_dict[video_name] = annos[video_name]
    video_list = list(video_dict.keys())
    video_list.sort()
    return video_dict, video_list


# Prediction
def infer_bmn(args):
    config = parse_config(args.config_file)
    infer_config = merge_configs(config, 'infer', vars(args))
    print_configs(infer_config, "Infer")

    if not os.path.isdir(infer_config.INFER.output_path):
        os.makedirs(infer_config.INFER.output_path)
    if not os.path.isdir(infer_config.INFER.result_path):
        os.makedirs(infer_config.INFER.result_path)

    place = paddle.CUDAPlace(0)
    paddle.disable_static(place)

    bmn = BMN(infer_config)
    # load checkpoint
    if args.weights:
        assert os.path.exists(
            args.weights +
            ".pdparams"), "Given weight dir {} not exist.".format(args.weights)

    logger.info('load test weights from {}'.format(args.weights))
    model_dict, _ = paddle.load(args.weights)
    bmn.set_dict(model_dict)

    infer_dataset = BmnDataset(infer_config, 'infer')
    infer_sampler = DistributedBatchSampler(
        infer_dataset, batch_size=infer_config.INFER.batch_size)
    infer_loader = DataLoader(
        infer_dataset,
        batch_sampler=infer_sampler,
        places=place,
        num_workers=infer_config.INFER.num_workers,
        return_list=True)

    video_dict, video_list = get_dataset_dict(infer_config)

    bmn.eval()
    for batch_id, data in enumerate(infer_loader):
        x_data = paddle.to_tensor(data[0])
        video_idx = data[1]  #batch_size=1 by default

        pred_bm, pred_start, pred_end = bmn(x_data)

        pred_bm = pred_bm.numpy()
        pred_start = pred_start[0].numpy()
        pred_end = pred_end[0].numpy()

        logger.info("Processing................ batch {}".format(batch_id))
        gen_props(
            pred_bm,
            pred_start,
            pred_end,
            video_idx,
            video_list,
            infer_config,
            mode='infer')

    logger.info("Post_processing....This may take a while")
    bmn_post_processing(video_dict, infer_config.INFER.subset,
                        infer_config.INFER.output_path,
                        infer_config.INFER.result_path)
    logger.info("[INFER] infer finished. Results saved in {}".format(
        args.save_dir) + "bmn_results_test.json")


if __name__ == '__main__':
    args = parse_args()
    infer_bmn(args)
