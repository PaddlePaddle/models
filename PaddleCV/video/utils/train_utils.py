#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import time
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import profiler
import logging
import shutil

logger = logging.getLogger(__name__)


def log_lr_and_step():
    try:
        # In optimizers, if learning_rate is set as constant, lr_var
        # name is 'learning_rate_0', and iteration counter is not 
        # recorded. If learning_rate is set as decayed values from 
        # learning_rate_scheduler, lr_var name is 'learning_rate', 
        # and iteration counter is recorded with name '@LR_DECAY_COUNTER@', 
        # better impliment is required here
        lr_var = fluid.global_scope().find_var("learning_rate")
        if not lr_var:
            lr_var = fluid.global_scope().find_var("learning_rate_0")
        lr = np.array(lr_var.get_tensor())

        lr_count = '[-]'
        lr_count_var = fluid.global_scope().find_var("@LR_DECAY_COUNTER@")
        if lr_count_var:
            lr_count = np.array(lr_count_var.get_tensor())
        logger.info("------- learning rate {}, learning rate counter {} -----"
                    .format(np.array(lr), np.array(lr_count)))
    except:
        logger.warn("Unable to get learning_rate and LR_DECAY_COUNTER.")


def test_with_dataloader(exe,
                         compiled_test_prog,
                         test_dataloader,
                         test_fetch_list,
                         test_metrics,
                         log_interval=0,
                         save_model_name=''):
    if not test_dataloader:
        logger.error("[TEST] get dataloader failed.")
    test_metrics.reset()
    test_iter = 0

    for data in test_dataloader():
        test_outs = exe.run(compiled_test_prog,
                            fetch_list=test_fetch_list,
                            feed=data)
        test_metrics.accumulate(test_outs)
        if log_interval > 0 and test_iter % log_interval == 0:
            test_metrics.calculate_and_log_out(test_outs, \
               info = '[TEST] test_iter {} '.format(test_iter))
        test_iter += 1
    test_metrics.finalize_and_log_out("[TEST] Finish")


def train_with_dataloader(exe, train_prog, compiled_train_prog, train_dataloader, \
                        train_fetch_list, train_metrics, epochs = 10, \
                        log_interval = 0, valid_interval = 0, save_dir = './', \
                        num_trainers = 1, trainer_id = 0, \
                        save_model_name = 'model', fix_random_seed = False, \
                        compiled_test_prog = None, test_dataloader = None, \
                        test_fetch_list = None, test_metrics = None, \
                        is_profiler = None, profiler_path = None):
    if not train_dataloader:
        logger.error("[TRAIN] get dataloader failed.")
    epoch_periods = []
    train_loss = 0
    for epoch in range(epochs):
        log_lr_and_step()

        train_iter = 0
        epoch_periods = []

        cur_time = time.time()
        for data in train_dataloader():
            train_outs = exe.run(compiled_train_prog,
                                 fetch_list=train_fetch_list,
                                 feed=data)
            period = time.time() - cur_time
            epoch_periods.append(period)
            timeStamp = time.time()
            localTime = time.localtime(timeStamp)
            strTime = time.strftime("%Y-%m-%d %H:%M:%S", localTime)
            if log_interval > 0 and (train_iter % log_interval == 0):
                train_metrics.calculate_and_log_out(train_outs, \
                        info = '[TRAIN {}] Epoch {}, iter {}, time {}, '.format(strTime, epoch, train_iter, period))
            train_iter += 1
            cur_time = time.time()

            # NOTE: profiler tools, used for benchmark
            if is_profiler and epoch == 0 and train_iter == log_interval:
                profiler.start_profiler("All")
            elif is_profiler and epoch == 0 and train_iter == log_interval + 5:
                profiler.stop_profiler("total", profiler_path)
                return

        if len(epoch_periods) < 1:
            logger.info(
                'No iteration was executed, please check the data reader')
            sys.exit(1)

        logger.info('[TRAIN] Epoch {} training finished, average time: {}'.
                    format(epoch, np.mean(epoch_periods[1:])))

        if trainer_id == 0:
            save_model(exe, train_prog, save_dir, save_model_name,
                       "_epoch{}".format(epoch))
        if compiled_test_prog and valid_interval > 0 and (
                epoch + 1) % valid_interval == 0:
            test_with_dataloader(exe, compiled_test_prog, test_dataloader,
                                 test_fetch_list, test_metrics, log_interval,
                                 save_model_name)

    if trainer_id == 0:
        save_model(exe, train_prog, save_dir, save_model_name)
    #when fix_random seed for debug
    if fix_random_seed:
        cards = os.environ.get('CUDA_VISIBLE_DEVICES')
        gpu_num = len(cards.split(","))
        print("kpis\ttrain_cost_card{}\t{}".format(gpu_num, train_loss))
        print("kpis\ttrain_speed_card{}\t{}".format(gpu_num,
                                                    np.mean(epoch_periods)))


def save_model(exe, program, save_dir, model_name, postfix=''):
    """save paramters and optimizer related varaibles"""
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    saved_model_name = model_name + postfix

    fluid.save(program, os.path.join(save_dir, saved_model_name))

    return
