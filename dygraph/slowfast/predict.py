#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import argparse
import ast
import logging
import numpy as np
import json

import paddle
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.hapi.model import _all_gather
import paddle.distributed as dist

from model import SlowFast
from save_load_helper import subn_load
from config_utils import parse_config, merge_configs, print_configs
from kinetics_dataset import KineticsDataset

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        "SLOWFAST test for performance evaluation.")
    parser.add_argument(
        '--config_file',
        type=str,
        default='slowfast.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='total eval batch size of all gpus.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--use_data_parallel',
        type=ast.literal_eval,
        default=True,
        help='default use data parallel.')
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='weight path, format as slowfast_epoch0.pdparams. None to use config setting.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='save path, None to use config setting.')
    args = parser.parse_args()
    return args


# Prediction
def infer_slowfast(args):
    config = parse_config(args.config_file)
    infer_config = merge_configs(config, 'infer', vars(args))
    print_configs(infer_config, "Infer")

    if not os.path.isdir(infer_config.INFER.save_path):
        os.makedirs(infer_config.INFER.save_path)

    place = 'gpu:{}'.format(dist.ParallelEnv()
                            .dev_id) if args.use_gpu else 'cpu'
    place = paddle.set_device(place)
    if args.use_data_parallel:
        dist.init_parallel_env()

    _nranks = dist.ParallelEnv().nranks  # num gpu
    assert _nranks == infer_config.INFER.num_gpus, \
        "num_gpus({}) set by CUDA_VISIBLE_DEVICES " \
        "shoud be the same as that " \
        "set in {}({})".format(
            _nranks, args.config, infer_config.INFER.num_gpus)
    bs_denominator = infer_config.INFER.num_gpus

    bs_single = int(infer_config.INFER.batch_size /
                    bs_denominator)  # batch_size of each gpu

    #build model
    slowfast = SlowFast(cfg=infer_config)
    # load checkpoint
    subn_load(slowfast, args.weights)

    if args.use_data_parallel:
        slowfast = paddle.DataParallel(slowfast)

    #create reader
    infer_data = KineticsDataset(mode="infer", cfg=infer_config)
    infer_sampler = DistributedBatchSampler(
        infer_data, batch_size=bs_single, shuffle=False, drop_last=False)
    infer_loader = DataLoader(
        infer_data,
        batch_sampler=infer_sampler,
        places=place,
        num_workers=infer_config.INFER.num_workers)

    # start infer
    num_ensemble_views = infer_config.INFER.num_ensemble_views
    num_spatial_crops = infer_config.INFER.num_spatial_crops
    num_cls = infer_config.DATA.num_classes
    num_clips = num_ensemble_views * num_spatial_crops
    num_videos = len(infer_data) // num_clips
    video_preds = np.zeros((num_videos, num_cls))
    clip_count = {}

    video_paths = []
    with open(infer_config.INFER.filelist, "r") as f:
        for path in f.read().splitlines():
            video_paths.append(path)

    print(
        "[INFER] infer start, number of videos {}, number of clips {}, total number of clips {}".
        format(num_videos, num_clips, num_clips * num_videos))
    slowfast.eval()
    for batch_id, data in enumerate(infer_loader):
        # call net
        model_inputs = [data[0], data[1]]
        preds = slowfast(model_inputs)
        clip_ids = data[3]

        # gather mulit card, results of following process in each card is the same.
        if _nranks > 1:
            preds = _all_gather(preds, _nranks)
            clip_ids = _all_gather(clip_ids, _nranks)

        # to numpy
        preds = preds.numpy()
        clip_ids = clip_ids.numpy()

        # preds ensemble
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // num_clips
            ts_idx = int(clip_ids[ind]) % num_clips
            if vid_id not in clip_count:
                clip_count[vid_id] = []
            if ts_idx in clip_count[vid_id]:
                print(
                    "[INFER] Passed!! read video {} clip index {} / {} repeatedly.".
                    format(vid_id, ts_idx, clip_ids[ind]))
            else:
                clip_count[vid_id].append(ts_idx)
                video_preds[vid_id] += preds[ind]  # ensemble method: sum
        if batch_id % args.log_interval == 0:
            print("[INFER] Processing batch {}/{} ...".format(
                batch_id, len(infer_data) // infer_config.INFER.batch_size))

    # check clip index of each video
    for key in clip_count.keys():
        if len(clip_count[key]) != num_clips or sum(clip_count[
                key]) != num_clips * (num_clips - 1) / 2:
            print(
                "[INFER] Warning!! video [{}] clip count [{}] not match number clips {}".
                format(key, clip_count[key], num_clips))

    res_list = []
    for j in range(video_preds.shape[0]):
        pred = paddle.to_tensor(video_preds[j] / num_clips)  #mean prob
        video_path = video_paths[j]
        pred = paddle.to_tensor(pred)
        top1_values, top1_indices = paddle.topk(pred, k=1)
        top5_values, top5_indices = paddle.topk(pred, k=5)
        top1_values = top1_values.numpy().astype("float64")[0]
        top1_indices = int(top1_indices.numpy()[0])
        top5_values = list(top5_values.numpy().astype("float64"))
        top5_indices = [int(item) for item in top5_indices.numpy()
                        ]  #np.int is not JSON serializable
        print("[INFER] video id [{}], top1 value {}, top1 indices {}".format(
            video_path, top1_values, top1_indices))
        print("[INFER] video id [{}], top5 value {}, top5 indices {}".format(
            video_path, top5_values, top5_indices))
        save_dict = {
            'video_id': video_path,
            'top1_values': top1_values,
            'top1_indices': top1_indices,
            'top5_values': top5_values,
            'top5_indices': top5_indices
        }
        res_list.append(save_dict)

    with open(
            os.path.join(infer_config.INFER.save_path, 'result' + '.json'),
            'w') as f:
        json.dump(res_list, f)
    print('[INFER] infer finished, results saved in {}'.format(
        infer_config.INFER.save_path))


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    infer_slowfast(args)
