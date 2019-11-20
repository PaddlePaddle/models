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


import os
import argparse


class Options:
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
        parser.add_argument('--aux', default=True,
                            help='Auxilary Loss')
        parser.add_argument('--se_loss', default=True,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--epoch_num', type=int, default=1200, metavar='N',
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
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr_scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--lr_pow', type=float, default=0.9,
                            help='learning rate scheduler (default: 0.9)')
        parser.add_argument('--lr_step', type=int, default=None,
                            help='lr step to change lr')
        parser.add_argument('--warm_up', type=bool, default=False,
                            help='warm_up (default: False)')
        parser.add_argument('--warmup_epoch', type=int, default=5,
                            help='warmup_epoch (default: 5)')
        parser.add_argument('--total_step', type=int, default=500000,
                            metavar='N', help='total_step (default: auto):500000)')
        parser.add_argument('--step_per_epoch', type=int, default=None,
                            metavar='N', help='step_per_epoch (default: auto)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=1e-4,   # 正则化系数
                            metavar='M', help='w-decay (default: 1e-4)')

        # cuda, seed and logging
        parser.add_argument('--cuda', default=True, type=bool,
                            help='use CUDA training')
        parser.add_argument('--use_data_parallel', default=True, type=bool,
                            help='use data_parallel training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log_root', type=str,
                            default='./', help='set a log path folder')

        # checkpoint
        parser.add_argument("--save_model", default='./checkpoint/', type=str,
                            help="model path")

        # finetuning pre-trained models
        parser.add_argument("--load_pretrained_model", default=True, type=bool,
                            help="load pretrained model (default: True)")

        # load better models
        parser.add_argument("--load_better_model", default=False, type=bool,
                            help="load better model (default: False)")

        parser.add_argument('--multi-scales', type=bool, default=True,
                            help="testing scale,default:(multi scale)")
        parser.add_argument('--flip', type=bool, default=True,
                            help="testing flip image,default:(True)")

        # multi grid dilation option
        parser.add_argument("--dilated", default=True, type=bool,
                            help="use dilation policy")
        parser.add_argument("--multi_grid", default=True, type=bool,
                            help="use multi grid dilation policy")
        parser.add_argument('--multi_dilation', type=int, default=[4, 8, 16],
                            help="multi grid dilation list")
        parser.add_argument('--scale', action='store_false', default=True,
                            help='choose to use random scale transform(0.75-2.0),default:multi scale')
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
                'cityscapes': 240,
            }
            num_class_dict = {
                'pascal_voc': None,
                'pascal_aug': None,
                'pcontext': None,
                'ade20k': None,
                'cityscapes': 19,
            }
            total_steps = {
                'pascal_voc': 500000,
                'pascal_aug': 500000,
                'pcontext': 500000,
                'ade20k': 500000,
                'cityscapes': 500000,
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
                'cityscapes': 185,  # 2975 // batch_size // GPU_num
            }
            args.step_per_epoch = step_per_epoch[args.dataset.lower()]
        if args.lr is None:
            lrs = {
                'pascal_voc': 0.0001,
                'pascal_aug': 0.001,
                'pcontext': 0.001,
                'ade20k': 0.01,
                'cityscapes': 0.01,
            }
            args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
        return args

    def print_args(self):
        arg_dict = self.parse().__dict__
        for k, v in arg_dict.items():
            print('{:30s}: {}'.format(k, v))

