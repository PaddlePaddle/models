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
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from model import TSM_ResNet
from config_utils import *
from reader import KineticsReader
from ucf101_reader import UCF101Reader
from timer import TimeAverager

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--config',
        type=str,
        default='tsm.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--use_data_parallel',
        type=ast.literal_eval,
        default=True,
        help='default use data parallel.')
    parser.add_argument(
        '--model_save_dir',
        type=str,
        default='./output',
        help='default model save in ./output.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--model_path_pre',
        type=str,
        default='tsm',
        help='default model path pre is tsm.')
    parser.add_argument(
        '--weights',
        type=str,
        default='./ResNet50_pretrained/',
        help='default weights is ./ResNet50_pretrained/.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')

    args = parser.parse_args()
    return args


def val(epoch, model, cfg, args):
    reader = UCF101Reader(name="TSM", mode="valid", cfg=cfg)
    reader = reader.create_reader()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0

    for batch_id, data in enumerate(reader()):
        x_data = np.array([item[0] for item in data])
        y_data = np.array([item[1] for item in data]).reshape([-1, 1])
        imgs = to_variable(x_data)
        labels = to_variable(y_data)
        labels.stop_gradient = True

        outputs = model(imgs)

        loss = fluid.layers.cross_entropy(
            input=outputs, label=labels, ignore_index=-1)
        avg_loss = fluid.layers.mean(loss)
        acc_top1 = fluid.layers.accuracy(input=outputs, label=labels, k=1)
        acc_top5 = fluid.layers.accuracy(input=outputs, label=labels, k=5)

        total_loss += avg_loss.numpy()[0]
        total_acc1 += acc_top1.numpy()[0]
        total_acc5 += acc_top5.numpy()[0]
        total_sample += 1

        print('TEST Epoch {}, iter {}, loss = {}, acc1 {}, acc5 {}'.format(
            epoch, batch_id,
            avg_loss.numpy()[0], acc_top1.numpy()[0], acc_top5.numpy()[0]))

    print('TEST Epoch {}, iter {}, Finish loss {} , acc1 {} , acc5 {}'.format(
        epoch, batch_id, total_loss / total_sample, total_acc1 / total_sample,
        total_acc5 / total_sample))


def create_optimizer(cfg, params):
    total_videos = cfg.total_videos
    step = int(total_videos / cfg.batch_size + 1)
    bd = [e * step for e in cfg.decay_epochs]
    base_lr = cfg.learning_rate
    lr_decay = cfg.learning_rate_decay
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    l2_weight_decay = cfg.l2_weight_decay
    momentum = cfg.momentum

    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=bd, values=lr),
        momentum=momentum,
        regularization=fluid.regularizer.L2Decay(l2_weight_decay),
        parameter_list=params)

    return optimizer


def train(args):
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')

    local_rank = fluid.dygraph.parallel.Env().local_rank

    use_data_parallel = args.use_data_parallel
    trainer_count = fluid.dygraph.parallel.Env().nranks
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        #(data_parallel step1/6)
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

    #load pretrain
    assert os.path.exists(args.weights), \
        "Given dir {} not exist.".format(args.weights)
    pre_state_dict = fluid.load_program_state(args.weights)
    #for key in pre_state_dict.keys():
    #    print('pre_state_dict.key: {}'.format(key))

    with fluid.dygraph.guard(place):
        #1. init model
        video_model = TSM_ResNet("TSM", train_config)

        #2. set weights
        param_state_dict = {}
        model_dict = video_model.state_dict()
        for key in model_dict.keys():
            weight_name = model_dict[key].name
            if weight_name in pre_state_dict.keys(
            ) and weight_name != "fc_0.w_0" and weight_name != "fc_0.b_0":
                print('succ Load weight: {}, shape: {}'.format(
                    weight_name, pre_state_dict[weight_name].shape))
                param_state_dict[key] = pre_state_dict[weight_name]
            else:
                print('fail Load weight: {}'.format(weight_name))
                param_state_dict[key] = model_dict[key]
        video_model.set_dict(param_state_dict)

        #3. init optim
        optimizer = create_optimizer(train_config.TRAIN,
                                     video_model.parameters())
        if use_data_parallel:
            #(data_parallel step2,3/6)
            strategy = fluid.dygraph.parallel.prepare_context()
            video_model = fluid.dygraph.parallel.DataParallel(video_model,
                                                              strategy)

        # 4. load checkpoint
        if args.checkpoint:
            assert os.path.exists(args.checkpoint + ".pdparams"), \
                "Given dir {}.pdparams not exist.".format(args.checkpoint)
            assert os.path.exists(args.checkpoint + ".pdopt"), \
                "Given dir {}.pdopt not exist.".format(args.checkpoint)
            para_dict, opti_dict = fluid.dygraph.load_dygraph(args.checkpoint)
            video_model.set_dict(para_dict)
            optimizer.set_dict(opti_dict)

        # 5. reader
        bs_denominator = 1
        if args.use_gpu:
            gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if gpus == "":
                pass
            else:
                gpus = gpus.split(",")
                num_gpus = len(gpus)
                assert num_gpus == train_config.TRAIN.num_gpus, \
                       "num_gpus({}) set by CUDA_VISIBLE_DEVICES" \
                       "shoud be the same as that" \
                       "set in {}({})".format(
                       num_gpus, args.config, train_config.TRAIN.num_gpus)
            bs_denominator = train_config.TRAIN.num_gpus

        train_config.TRAIN.batch_size = int(train_config.TRAIN.batch_size /
                                            bs_denominator)

        train_reader = UCF101Reader(name="TSM", mode="train", cfg=train_config)

        train_reader = train_reader.create_reader()
        if use_data_parallel:
            #(data_parallel step4/6)
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        # 6. train loop
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        for epoch in range(train_config.TRAIN.epoch):
            epoch_start = time.time()

            video_model.train()
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0

            # 6.1 for each batch, call model() , backward(), and minimize()
            batch_start = time.time()
            for batch_id, data in enumerate(train_reader()):
                t1 = time.time()
                reader_cost_averager.record(t1 - batch_start)

                x_data = np.array([item[0] for item in data])
                y_data = np.array([item[1] for item in data]).reshape([-1, 1])

                imgs = to_variable(x_data)
                labels = to_variable(y_data)
                labels.stop_gradient = True

                t2 = time.time()
                outputs = video_model(imgs)
                t3 = time.time()

                loss = fluid.layers.cross_entropy(
                    input=outputs, label=labels, ignore_index=-1)
                avg_loss = fluid.layers.mean(loss)

                acc_top1 = fluid.layers.accuracy(
                    input=outputs, label=labels, k=1)
                acc_top5 = fluid.layers.accuracy(
                    input=outputs, label=labels, k=5)

                current_step_lr = optimizer.current_step_lr()
                if use_data_parallel:
                    #(data_parallel step5/6)
                    avg_loss = video_model.scale_loss(avg_loss)
                    avg_loss.backward()
                    video_model.apply_collective_grads()
                else:
                    avg_loss.backward()

                t4 = time.time()
                optimizer.minimize(avg_loss)
                video_model.clear_gradients()

                avg_loss_value = avg_loss.numpy()[0]
                acc_top1_value = acc_top1.numpy()[0]
                acc_top5_value = acc_top5.numpy()[0]

                total_loss += avg_loss_value
                total_acc1 += acc_top1_value
                total_acc5 += acc_top5_value
                total_sample += 1

                t5 = time.time()
                batch_cost_averager.record(
                    t5 - batch_start, num_samples=train_config.TRAIN.batch_size)
                if batch_id % args.log_interval == 0:
                    print(
                        'TRAIN Epoch: %d, iter: %d, loss: %.5f, acc1: %.5f, acc5: %.5f, lr: %.5f, forward_cost:%.5f s, backward_cost:%.5f s, minimize_cost:%.5f s, to_variable_cost: %.5f s, batch_cost: %.5f sec, reader_cost: %.5f sec, ips: %.5f samples/sec'
                        % (epoch, batch_id, avg_loss_value, acc_top1_value,
                           acc_top5_value, current_step_lr, t3 - t2, t4 - t3,
                           t5 - t4, t2 - t1, batch_cost_averager.get_average(),
                           reader_cost_averager.get_average(),
                           batch_cost_averager.get_ips_average()))
                    batch_cost_averager.reset()
                    reader_cost_averager.reset()

                batch_start = time.time()

            train_epoch_cost = time.time() - epoch_start
            print(
                'TRAIN End, Epoch {}, avg_loss= {:.5f}, avg_acc1= {:.5f}, avg_acc5= {:.5f}, lr={:.5f}, epoch_cost: {:.5f} sec'.
                format(epoch, total_loss / total_sample, total_acc1 /
                       total_sample, total_acc5 / total_sample, current_step_lr,
                       train_epoch_cost))

            # 6.2 save checkpoint 
            if local_rank == 0:
                if not os.path.isdir(args.model_save_dir):
                    os.makedirs(args.model_save_dir)
                model_path = os.path.join(
                    args.model_save_dir,
                    args.model_path_pre + "_epoch{}".format(epoch))
                fluid.dygraph.save_dygraph(video_model.state_dict(), model_path)
                fluid.dygraph.save_dygraph(optimizer.state_dict(), model_path)
                print('save_dygraph End, Epoch {}/{} '.format(
                    epoch, train_config.TRAIN.epoch))

            # 6.3 validation
            video_model.eval()
            val(epoch, video_model, valid_config, args)

        # 7. save final model
        if local_rank == 0:
            model_path = os.path.join(args.model_save_dir,
                                      args.model_path_pre + "_final")
            fluid.dygraph.save_dygraph(video_model.state_dict(), model_path)
            fluid.dygraph.save_dygraph(optimizer.state_dict(), model_path)

        logger.info('[TRAIN] training finished')


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    train(args)
