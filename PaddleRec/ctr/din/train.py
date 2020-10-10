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
import logging
import time
import numpy as np
import argparse
import paddle.fluid as fluid
import paddle
import time
import network
import reader
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("din")
    parser.add_argument(
        '--config_path',
        type=str,
        default='data/config.txt',
        help='dir of config')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='data/paddle_train.txt',
        help='dir of train file')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='din_amazon',
        help='dir of saved model')
    parser.add_argument(
        '--batch_size', type=int, default=16, help='number of batch size')
    parser.add_argument(
        '--epoch_num', type=int, default=200, help='number of epoch')
    parser.add_argument(
        '--use_cuda', type=int, default=0, help='whether to use gpu')
    parser.add_argument(
        '--parallel',
        type=int,
        default=0,
        help='whether to use parallel executor')
    parser.add_argument(
        '--base_lr', type=float, default=0.85, help='based learning rate')
    parser.add_argument(
        '--num_devices', type=int, default=1, help='Number of GPU devices')
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help='If set, run the task with continuous evaluation logs.')
    parser.add_argument('--batch_num', type=int, help="batch num for ce")
    args = parser.parse_args()
    return args


def train():
    args = parse_args()

    if args.enable_ce:
        SEED = 102
        fluid.default_main_program().random_seed = SEED
        fluid.default_startup_program().random_seed = SEED

    config_path = args.config_path
    train_path = args.train_dir
    epoch_num = args.epoch_num
    use_cuda = True if args.use_cuda else False
    use_parallel = True if args.parallel else False

    logger.info("reading data begins")
    user_count, item_count, cat_count = reader.config_read(config_path)
    data_reader, max_len = reader.prepare_reader(train_path, args.batch_size *
                                                 args.num_devices)
    logger.info("reading data completes")

    avg_cost, pred, feed_list = network.network(item_count, cat_count)

    clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0)
    base_lr = args.base_lr
    boundaries = [410000]
    values = [base_lr, 0.2]
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values),
        grad_clip=clip)
    sgd_optimizer.minimize(avg_cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    loader = fluid.io.DataLoader.from_generator(
        feed_list=feed_list, capacity=10000, iterable=True)
    loader.set_sample_list_generator(data_reader, places=place)
    if use_parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda, loss_name=avg_cost.name)
    else:
        train_exe = exe

    logger.info("train begins")

    global_step = 0
    PRINT_STEP = 1000

    total_time = []
    ce_info = []
    start_time = time.time()
    loss_sum = 0.0
    for id in range(epoch_num):
        epoch = id + 1
        for data in loader():
            global_step += 1
            results = train_exe.run(feed=data,
                                    fetch_list=[avg_cost.name, pred.name],
                                    return_numpy=True)
            loss_sum += results[0].mean()

            if global_step % PRINT_STEP == 0:
                ce_info.append(loss_sum / PRINT_STEP)
                total_time.append(time.time() - start_time)
                logger.info(
                    "epoch: %d\tglobal_step: %d\ttrain_loss: %.4f\t\ttime: %.2f"
                    % (epoch, global_step, loss_sum / PRINT_STEP,
                       time.time() - start_time))
                start_time = time.time()
                loss_sum = 0.0

                if (global_step > 400000 and global_step % PRINT_STEP == 0) or (
                        global_step <= 400000 and global_step % 50000 == 0):
                    save_dir = os.path.join(args.model_dir,
                                            "global_step_" + str(global_step))
                    feed_var_name = [
                        "hist_item_seq", "hist_cat_seq", "target_item",
                        "target_cat", "label", "mask", "target_item_seq",
                        "target_cat_seq"
                    ]
                    fetch_vars = [avg_cost, pred]
                    fluid.io.save_inference_model(save_dir, feed_var_name,
                                                  fetch_vars, exe)
                    logger.info("model saved in " + save_dir)
            if args.enable_ce and global_step >= args.batch_num:
                break
    # only for ce
    if args.enable_ce:
        gpu_num = get_cards(args)
        ce_loss = 0
        ce_time = 0
        try:
            ce_loss = ce_info[-1]
            ce_time = total_time[-1]
        except:
            print("ce info error")
        print("kpis\teach_pass_duration_card%s\t%s" % (gpu_num, ce_time))
        print("kpis\ttrain_loss_card%s\t%s" % (gpu_num, ce_loss))


def get_cards(args):
    if args.enable_ce:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        num = len(cards.split(","))
        return num
    else:
        return args.num_devices


if __name__ == "__main__":
    import paddle
    paddle.enable_static()
    train()
