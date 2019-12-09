# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Paddle DANet Segmentation')

        # model and dataset
        parser.add_argument('--model', type=str, default='danet',
                            help='model name (default: danet)')
        parser.add_argument('--backbone', type=str, default='resnet101',
                            help='backbone name (default: resnet101)')
        parser.add_argument('--dataset', type=str, default='cityscapes',
                            help='dataset name (default: cityscapes)')
        parser.add_argument('--num_classes', type=int, default=19,
                            help='num_classes (default: cityscapes = 19)')
        parser.add_argument('--data_folder', type=str,
                            default='./dataset',
                            help='training dataset folder (default: ./dataset')
        parser.add_argument('--base_size', type=int, default=1024,
                            help='base image size')
        parser.add_argument('--crop_size', type=int, default=768,
                            help='crop image size')

        # training hyper params
        parser.add_argument('--epoch_num', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch_size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test_batch_size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')

        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr_scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--lr_pow', type=float, default=0.9,
                            help='learning rate scheduler (default: 0.9)')
        parser.add_argument('--lr_step', type=int, default=None,
                            help='lr step to change lr')
        parser.add_argument('--warm_up', action='store_true', default=False,
                            help='warm_up (default: False)')
        parser.add_argument('--warmup_epoch', type=int, default=5,
                            help='warmup_epoch (default: 5)')
        parser.add_argument('--total_step', type=int, default=None,
                            metavar='N', help='total_step (default: auto)')
        parser.add_argument('--step_per_epoch', type=int, default=None,
                            metavar='N', help='step_per_epoch (default: auto)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=1e-4,   
                            metavar='M', help='w-decay (default: 1e-4)')

        # cuda, seed and logging
        parser.add_argument('--cuda', action='store_true', default=False,
                            help='use CUDA training, (default: False)')
        parser.add_argument('--use_data_parallel', action='store_true', default=False,
                            help='use data_parallel training, (default: False)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log_root', type=str,
                            default='./', help='set a log path folder')

        # checkpoint
        parser.add_argument("--save_model", default='checkpoint/DANet101_better_model_paddle1.6', type=str,
                            help="model path, (default: checkpoint/DANet101_better_model_paddle1.6)")
        
        # change executor model params to dygraph model params
        parser.add_argument("--change_executor_to_dygraph", action='store_true', default=False,
                            help="change executor model params to dygraph model params (default:False)")

        # finetuning pre-trained models
        parser.add_argument("--load_pretrained_model", action='store_true', default=False,
                            help="load pretrained model (default: False)")
        # load better models
        parser.add_argument("--load_better_model", action='store_true', default=False,
                            help="load better model (default: False)")
        parser.add_argument('--multi_scales', action='store_true', default=False,
                            help="testing scale, (default: False)")
        parser.add_argument('--flip', action='store_true', default=False,
                            help="testing flip image, (default: False)")

        # multi grid dilation option
        parser.add_argument("--dilated", action='store_true', default=False,
                            help="use dilation policy, (default: False)")
        parser.add_argument("--multi_grid", action='store_true', default=False,
                            help="use multi grid dilation policy, default: False")
        parser.add_argument('--multi_dilation', nargs='+', type=int, default=None,
                            help="multi grid dilation list, (default: None), can use --mutil_dilation 4 8 16")
        parser.add_argument('--scale', action='store_true', default=False,
                            help='choose to use random scale transform(0.75-2.0) for train, (default: False)')
        
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs, batch_size and lr
        if args.epoch_num is None:
            epoches = {
                'pascal_voc': 180,
                'pascal_aug': 180,
                'pcontext': 180,
                'ade20k': 180,
                'cityscapes': 350,
            }
            num_class_dict = {
                'pascal_voc': 21,
                'pascal_aug': 21,
                'pcontext': 21,
                'ade20k': None,
                'cityscapes': 19,
            }
            total_steps = {
                'pascal_voc': 200000,
                'pascal_aug': 500000,
                'pcontext': 500000,
                'ade20k': 500000,
                'cityscapes': 150000,
            }
            args.epoch_num = epoches[args.dataset.lower()]
            args.num_classes = num_class_dict[args.dataset.lower()]
            args.total_step = total_steps[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 2
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.step_per_epoch is None:
            step_per_epoch = {
                'pascal_voc': 185,
                'pascal_aug': 185,
                'pcontext': 185,
                'ade20k': 185,
                'cityscapes': 371,  # 2975 // batch_size // GPU_num
            }
            args.step_per_epoch = step_per_epoch[args.dataset.lower()]
        if args.lr is None:
            lrs = {
                'pascal_voc': 0.0001,
                'pascal_aug': 0.001,
                'pcontext': 0.001,
                'ade20k': 0.01,
                'cityscapes': 0.003,
            }
            args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
        return args

    def print_args(self):
        arg_dict = self.parse().__dict__
        for k, v in arg_dict.items():
            print('{:30s}: {}'.format(k, v))

