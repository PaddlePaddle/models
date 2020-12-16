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
        help='Weight path, format as slowfast_epoch0, without suffix. None to use config setting.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


# Performance Evaluation
def test_slowfast(args):
    config = parse_config(args.config_file)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, "Test")

    place = 'gpu:{}'.format(dist.ParallelEnv()
                            .dev_id) if args.use_gpu else 'cpu'
    place = paddle.set_device(place)
    if args.use_data_parallel:
        dist.init_parallel_env()

    _nranks = dist.ParallelEnv().nranks  # num gpu
    assert _nranks == test_config.TEST.num_gpus, \
        "num_gpus({}) set by CUDA_VISIBLE_DEVICES " \
        "shoud be the same as that " \
        "set in {}({})".format(
            _nranks, args.config_file, test_config.TEST.num_gpus)
    bs_denominator = test_config.TEST.num_gpus

    bs_single = int(test_config.TEST.batch_size /
                    bs_denominator)  # batch_size of each gpu

    # build model
    slowfast = SlowFast(cfg=test_config)
    # load checkpoint
    subn_load(slowfast, args.weights)

    if args.use_data_parallel:
        slowfast = paddle.DataParallel(slowfast)

    #create reader
    test_data = KineticsDataset(mode="test", cfg=test_config)
    test_sampler = DistributedBatchSampler(
        test_data, batch_size=bs_single, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test_data,
        batch_sampler=test_sampler,
        places=place,
        num_workers=test_config.TEST.num_workers)

    # start eval
    num_ensemble_views = test_config.TEST.num_ensemble_views
    num_spatial_crops = test_config.TEST.num_spatial_crops
    num_cls = test_config.DATA.num_classes
    num_clips = num_ensemble_views * num_spatial_crops
    num_videos = len(test_data) // num_clips
    video_preds = np.zeros((num_videos, num_cls))
    video_labels = np.zeros((num_videos, 1), dtype="int64")
    clip_count = {}

    print("[EVAL] eval start, number of videos {}, total number of clips {}".
          format(num_videos, num_clips * num_videos))
    slowfast.eval()
    for batch_id, data in enumerate(test_loader):
        # call net
        model_inputs = [data[0], data[1]]
        preds = slowfast(model_inputs)
        labels = data[2]
        clip_ids = data[3]

        # gather mulit card, results of following process in each card is the same.
        if _nranks > 1:
            preds = _all_gather(preds, _nranks)
            labels = _all_gather(labels, _nranks)
            clip_ids = _all_gather(clip_ids, _nranks)

        # to numpy
        preds = preds.numpy()
        labels = labels.numpy().astype("int64")
        clip_ids = clip_ids.numpy()

        # preds ensemble
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // num_clips
            ts_idx = int(clip_ids[ind]) % num_clips
            if vid_id not in clip_count:
                clip_count[vid_id] = []
            if ts_idx in clip_count[vid_id]:
                print(
                    "[EVAL] Passed!! read video {} clip index {} / {} repeatedly.".
                    format(vid_id, ts_idx, clip_ids[ind]))
            else:
                clip_count[vid_id].append(ts_idx)
                video_preds[vid_id] += preds[ind]  # ensemble method: sum
                if video_labels[vid_id].sum() > 0:
                    assert video_labels[vid_id] == labels[ind]
                video_labels[vid_id] = labels[ind]
        if batch_id % args.log_interval == 0:
            print("[EVAL] Processing batch {}/{} ...".format(
                batch_id, len(test_data) // test_config.TEST.batch_size))

    # check clip index of each video
    for key in clip_count.keys():
        if len(clip_count[key]) != num_clips or sum(clip_count[
                key]) != num_clips * (num_clips - 1) / 2:
            print(
                "[EVAL] Warning!! video [{}] clip count [{}] not match number clips {}".
                format(key, clip_count[key], num_clips))

    video_preds = paddle.to_tensor(video_preds)
    video_labels = paddle.to_tensor(video_labels)
    acc_top1 = paddle.metric.accuracy(
        input=video_preds, label=video_labels, k=1)
    acc_top5 = paddle.metric.accuracy(
        input=video_preds, label=video_labels, k=5)
    print('[EVAL] eval finished, avg_acc1= {}, avg_acc5= {} '.format(
        acc_top1.numpy(), acc_top5.numpy()))


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    test_slowfast(args)
