# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 
Copy-paste from PaddleSeg with minor modifications.
https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/paddleseg/core/train.py
"""

import os
import time

import numpy as np
import paddle
import paddle.nn.functional as F

from smoke.utils import TimeAverager, calculate_eta, logger, progbar
from .kitti_eval import kitti_evaluation


def evaluate(model,
             eval_dataset,
             num_workers=0,
             output_dir="./output",
             print_detail=True):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
  
    batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)

    if print_detail:
        logger.info(
            "Start evaluating (total_samples={}, total_iters={})...".format(
                len(eval_dataset), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    predictions = {}
    with paddle.no_grad():
        for cur_iter, batch in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            images, targets, image_ids = batch[0], batch[1], batch[2]

            output = model(images, targets)
            
            output = output.numpy()
            predictions.update(
                {img_id: output for img_id in image_ids})
            
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(targets))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if print_detail:
                progbar_val.update(cur_iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    kitti_evaluation(eval_dataset, predictions, output_dir=output_dir)

    
