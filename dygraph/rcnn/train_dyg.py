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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import re
import numpy as np
import time
import shutil
import collections
import paddle.fluid as fluid
import reader
from models.dyg.model_builder import RCNN
from config import cfg
from utility import parse_args, print_arguments, SmoothedValue, TrainingStats, now_time, check_gpu

num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 2))


def get_device_num():
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1:
        return 1
    return fluid.core.get_cuda_device_count()


def optimizer_setting(params_list, gpu_num=8):
    base_lr = cfg.learning_rate / 8.0 * gpu_num  # * cfg.TRAIN.im_per_batch
    boundaries = []
    values = []

    # exponential warm up
    warmup_steps = cfg.warm_up_iter
    for i in range(warmup_steps):
        alpha = i / warmup_steps
        factor = cfg.warm_up_factor * (1 - alpha) + alpha
        lr = base_lr * factor
        values.append(lr)
        if i > 0:
            # because step_num start from 1
            boundaries.append(i)

    boundaries.append(warmup_steps)
    values.append(base_lr)

    # picewise decay
    boundaries.extend(cfg.lr_steps)
    values.extend([
        cfg.learning_rate * (cfg.lr_gamma**(i + 1))
        for i in range(len(cfg.lr_steps))
    ])

    decay_lr = fluid.dygraph.PiecewiseDecay(
        boundaries=boundaries, values=values, begin=0, step=1)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=decay_lr,
        parameter_list=params_list,
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)

    return optimizer, decay_lr


def train():
    if cfg.enable_ce:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        import random
        random.seed(0)
        np.random.seed(0)

    devices_num = 1
    total_batch_size = devices_num * cfg.TRAIN.im_per_batch

    use_random = True
    if cfg.enable_ce:
        use_random = False

    if cfg.parallel:
        strategy = fluid.dygraph.parallel.prepare_context()
        print("Execute Parallel Mode!!!")

    # Model
    model = RCNN("faster_rcnn", cfg=cfg, mode='train', use_random=use_random)

    # Optimizer
    optimizer, lr_scheduler = optimizer_setting(model.parameters(), gpu_num=8)

    if cfg.parallel:
        model = fluid.dygraph.parallel.DataParallel(model, strategy)

    if cfg.pretrained_model:
        model_state = model.state_dict()
        w_dict = np.load(cfg.pretrained_model)
        for k, v in w_dict.items():
            for wk in model_state.keys():
                res = re.search(k, wk)
                if res is not None:
                    print("load: ", k, v.shape, np.mean(np.abs(v)), " --> ", wk,
                          model_state[wk].shape)
                    model_state[wk] = v
                    break
        model.set_dict(model_state)

    elif cfg.resume_model:
        para_state_dict, _ = fluid.load_dygraph(cfg.resume_model)
        model.set_dict(para_state_dict)

        new_dict = {}
        for k, v in para_state_dict.items():
            if "conv2d" in k:
                new_k = k.split('.')[1]
            elif 'linear' in k:
                new_k = k.split('.')[1]
            elif 'conv2dtranspose' in k:
                new_k = k.split('.')[1]
            else:
                new_k = k
            new_dict[new_k] = v.numpy()
        np.savez("rcnn_dyg.npz", **new_dict)

    shuffle = True
    if cfg.enable_ce:
        shuffle = False
    # NOTE: do not shuffle dataset when using multi-process training
    shuffle_seed = None
    if num_trainers > 1:
        shuffle_seed = 1

    train_reader = reader.train(batch_size=total_batch_size, shuffle=shuffle)
    if cfg.parallel:
        train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)

    def save_model(model_state, postfix):
        file_path = os.path.join(cfg.model_save_dir, postfix)
        fluid.dygraph.save_dygraph(model_state, file_path)

    def train_loop():
        start_time = time.time()
        prev_start_time = start_time
        start = start_time
        train_stats = None  #TrainingStats(cfg.log_window, keys)

        for iter_id, data in enumerate(train_reader()):

            if iter_id == 100:
                print("=====Start Record Time=====")
                srt = time.time()

            if cfg.enable_ce:
                if iter_id == 10:
                    break

            prev_start_time = start_time
            start_time = time.time()

            model.train()
            # im, gt_boxes, gt_classes, is_crowd, im_info, im_id, gt_masks
            gt_max_num = 0
            poly_max_num = 0
            point_max_num = 0
            batch_size = len(data)
            x = data[0]

            # batch
            for x in data:
                if x[1].shape[0] > gt_max_num:
                    gt_max_num = x[1].shape[0]

                if cfg.MASK_ON:
                    for p in x[-1]:
                        for i, pp in enumerate(p):
                            if (i + 1) > poly_max_num:
                                poly_max_num = i + 1
                            if pp.shape[0] > point_max_num:
                                point_max_num = pp.shape[0]

            gt_box_data = np.zeros([batch_size, gt_max_num, 4])
            gt_label_data = np.zeros([batch_size, gt_max_num])
            is_crowd_data = np.ones([batch_size, gt_max_num])
            if cfg.MASK_ON:
                gt_masks_data = -np.ones(
                    [batch_size, gt_max_num, poly_max_num, point_max_num, 2])

            for batch_id, x in enumerate(data):
                gt_num = x[1].shape[0]
                gt_box_data[batch_id, 0:gt_num, :] = x[1]
                gt_label_data[batch_id, 0:gt_num] = x[2]
                is_crowd_data[batch_id, 0:gt_num] = x[3]
                if cfg.MASK_ON:
                    # all masks of one image
                    for i, gt_seg in enumerate(x[-1]):
                        for ii, poly in enumerate(gt_seg):
                            gt_masks_data[batch_id, i, ii, :poly.shape[
                                0], :] = poly

            gt_box_data = gt_box_data.astype('float32')
            gt_label_data = gt_label_data.astype('int32')
            is_crowd_data = is_crowd_data.astype('int32')
            if cfg.MASK_ON:
                gt_masks_data = gt_masks_data.astype("float32")

            image_data = np.array([x[0] for x in data]).astype('float32')
            im_info_data = np.array([x[4] for x in data]).astype('float32')
            im_id_data = np.array([x[5] for x in data]).astype('int32')

            if cfg.enable_ce:
                print("image_data: ", np.abs(image_data).mean(),
                      image_data.shape)
                print("gt_boxes: ", np.abs(gt_box_data).mean(),
                      gt_box_data.shape)
                print("gt_classes: ", np.abs(gt_label_data).mean(),
                      gt_label_data.shape)
                print("is_crowd: ", np.abs(is_crowd_data).mean(),
                      is_crowd_data.shape)
                print("im_info_dta: ", np.abs(im_info_data).mean(),
                      im_info_data.shape, im_info_data)
                print("img_id: ", im_id_data, im_id_data.shape)
                if cfg.MASK_ON:
                    print("gt_masks: ", np.abs(gt_masks_data).mean(),
                          gt_masks_data.shape)
            padding_time = time.time()

            # forward
            outputs = model(image_data, im_info_data, gt_box_data,
                            gt_label_data, is_crowd_data, gt_masks_data
                            if cfg.MASK_ON else None)

            # backward
            loss = outputs['loss']
            if cfg.parallel:
                loss = model.scale_loss(loss)
                loss.backward()
                model.apply_collective_grads()
            else:
                loss.backward()
            optimizer.minimize(loss)
            model.clear_gradients()

            run_model_time = time.time()

            if iter_id == 200:
                print("=====Avg Run Time %s=====" % (
                    (time.time() - srt) / 100.0))

            # Print train state
            keys = list([k for k in outputs.keys() if 'loss' in k])
            if train_stats is None:
                #print("debug log_window: ", cfg.log_window)
                train_stats = TrainingStats(cfg.log_window, keys)
            lr = lr_scheduler.get_learning_rate()

            losses = list(outputs[k] for k in keys)
            losses_np = [l.numpy() for l in losses]
            stats = {k: np.array(v).mean() for k, v in zip(keys, losses_np)}
            train_stats.update(stats)
            logs = train_stats.log()
            strs = '{}, iter: {}, lr: {}, {}, time: {:.3f}'.format(
                now_time(), iter_id, lr, logs, start_time - prev_start_time)
            print(strs)
            last_time = time.time()
            pad_t = padding_time - start_time
            run_t = run_model_time - padding_time
            print_t = last_time - run_model_time
            data_t = last_time - start_time - pad_t - run_t - print_t
            sys.stdout.flush()
            if (iter_id) % cfg.TRAIN.snapshot_iter == 0:
                save_model(model.state_dict(), 'model_iter%d' % (iter_id))
                save_model(optimizer.state_dict(), 'optim_iter%d' % (iter_id))
            if (iter_id + 1) == cfg.max_iter:
                break
        end_time = time.time()
        total_time = end_time - start_time

    train_loop()
    save_model(model.state_dict(), 'model_final')
    save_model(optimizer.state_dict(), 'optim_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if cfg.parallel else fluid.CUDAPlace(0) \
        if cfg.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        train()
