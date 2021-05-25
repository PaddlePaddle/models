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
from collections import deque
import shutil

import paddle
import paddle.nn.functional as F
from visualdl import LogWriter

from smoke.utils import TimeAverager, calculate_eta, logger

def train(model,
          train_dataset,
          val_dataset=None,
          optimizer=None,
          loss_computation=None,
          save_dir='output',
          iters=10000,
          batch_size=2,
          resume_model=None,
          save_interval=1000,
          log_iters=10,
          num_workers=0,
          keep_checkpoint_max=5):
    """
    Launch training.

    Args:
        model（nn.Layer): A sementic segmentation model.
        train_dataset (paddle.io.Dataset): Used to read and process training datasets.
        val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
        optimizer (paddle.optimizer.Optimizer): The optimizer.
        loss_computation (nn.Layer): A loss function.
        save_dir (str, optional): The directory for saving the model snapshot. Default: 'output'.
        iters (int, optional): How may iters to train the model. Defualt: 10000.
        batch_size (int, optional): Mini batch size of one gpu or cpu. Default: 2.
        resume_model (str, optional): The path of resume model.
        save_interval (int, optional): How many iters to save a model snapshot once during training. Default: 1000.
        log_iters (int, optional): Display logging information at every log_iters. Default: 10.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        keep_checkpoint_max (int, optional): Maximum number of checkpoints to save. Default: 5.
    """
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    start_iter = 0
    if resume_model is not None:
        start_iter = resume(model, optimizer, resume_model)

    if not os.path.isdir(save_dir):
        if os.path.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)

    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
            ddp_model = paddle.DataParallel(model)
        else:
            ddp_model = paddle.DataParallel(model)

    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    # VisualDL log
    log_writer = LogWriter(save_dir)

    avg_loss = 0.0
    avg_loss_dict = {}
    iters_per_epoch = len(batch_sampler)

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()

    iter = start_iter
    while iter < iters:
        for data in loader:
            iter += 1
            if iter > iters:
                break
            reader_cost_averager.record(time.time() - batch_start)
            images = data[0]
            targets = data[1]

            if nranks > 1:
                predictions = ddp_model(images)
            else:
                predictions = model(images)

            loss_dict = loss_computation(predictions, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()

            optimizer.step()
            lr = optimizer.get_lr()
            if isinstance(optimizer._learning_rate,
                          paddle.optimizer.lr.LRScheduler):
                optimizer._learning_rate.step()
            model.clear_gradients()
            avg_loss += loss.numpy()[0] # get the value
            if len(avg_loss_dict) == 0:
                avg_loss_dict = {k:v.numpy()[0] for k, v in loss_dict.items()}
            else:
                for key, value in loss_dict.items():
                    avg_loss_dict[key] += value.numpy()[0]

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)

            if (iter) % log_iters == 0 and local_rank == 0:
                avg_loss /= log_iters
                for key, value in avg_loss_dict.items():
                    avg_loss_dict[key] /= log_iters

                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
                logger.info(
                    "[TRAIN] epoch={}, iter={}/{}, loss={:.4f}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.5f} | ETA {}"
                    .format((iter - 1) // iters_per_epoch + 1, iter, iters,
                            avg_loss, lr, avg_train_batch_cost,
                            avg_train_reader_cost, eta))
                
                ######################### VisualDL Log ##########################
                log_writer.add_scalar('Train/loss', avg_loss, iter)
                # Record all losses if there are more than 2 losses.
                for key, value in avg_loss_dict.items():
                    log_tag = 'Train/' + key
                    log_writer.add_scalar(log_tag, value, iter)

                log_writer.add_scalar('Train/lr', lr, iter)
                log_writer.add_scalar('Train/batch_cost',
                                        avg_train_batch_cost, iter)
                log_writer.add_scalar('Train/reader_cost',
                                        avg_train_reader_cost, iter)
                #################################################################

                avg_loss = 0.0
                avg_loss_list = {}
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if (iter % save_interval == 0 or iter == iters) and local_rank == 0:
                current_save_dir = os.path.join(save_dir,
                                                "iter_{}".format(iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))
                save_models.append(current_save_dir)
                if len(save_models) > keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                
            batch_start = time.time()


    # Sleep for half a second to let dataloader release resources.
    time.sleep(0.5)
    log_writer.close()