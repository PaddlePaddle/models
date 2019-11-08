# -*- encoding: utf-8 -*-
# Software: PyCharm
# Time    : 2019/9/15 
# Author  : Wang
# File    : options.py


import os
import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Paddle DANet Segmentation')

        # model and dataset
        parser.add_argument('--model', type=str, default='danet',
                            help='model name (default: danet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='cityscapes',
                            help='dataset name (default: cityscapes)')
        parser.add_argument('--num_classes', type=int, default=19,
                            help='num_classes (default: cityscapes = 19)')
        parser.add_argument('--data_folder', type=str,
                            default='./dataset',
                            help='training dataset folder (default: ./dataset')
        parser.add_argument('--base_size', type=int, default=228, # 1000
                            help='base image size')
        parser.add_argument('--crop_size', type=int, default=224, # 740
                            help='crop image size')

        # training hyper params
        parser.add_argument('--aux', action='store_true', default=False,
                            help='Auxilary Loss')
        parser.add_argument('--se_loss', action='store_true', default=False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--epoch_num', type=int, default=1950, metavar='N',
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
                            help='warm_up (default: True)')
        parser.add_argument('--warmup_epoch', type=int, default=5,
                            help='warmup_epoch (default: 5)')
        parser.add_argument('--total_step', type=int, default=450000,
                            metavar='N', help='total_step (default: auto):450000)')
        parser.add_argument('--step_per_epoch', type=int, default=None,
                            metavar='N', help='step_per_epoch (default: auto)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=1e-5,
                            metavar='M', help='w-decay (default: 1e-5)')

        # cuda, seed and logging
        parser.add_argument('--cuda', default=True, type=bool,
                            help='use CUDA training')
        parser.add_argument('--use_data_parallel', default=False, type=bool,
                            help='use data_parallel training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log_root', type=str,
                            default='./', help='set a log path folder')

        # checkpoint
        parser.add_argument("--save_model", default='./checkpoint/', type=str,
                            help="model path")
        # ---------------------------------------下面暂时未用到------------------------------------
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--resume-dir', type=str, default=None,
                            help='put the path to resuming dir if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default=False,
                            help='finetuning on a different dataset')
        parser.add_argument('--ft-resume', type=str, default=None,
                            help='put the path of trained model to finetune if needed ')
        parser.add_argument('--pre-class', type=int, default=None,
                            help='num of pre-trained classes \
                            (default: None)')

        # evaluation option
        parser.add_argument('--ema', action='store_true', default=False,
                            help='using EMA evaluation')
        parser.add_argument('--eval', action='store_true', default=False,
                            help='evaluating mIoU')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        parser.add_argument('--multi-scales', type=bool, default=True,
                            help="testing scale,default:(multi scale)")
        parser.add_argument('--flip', type=bool, default=True,
                            help="testing flip image,default:(True)")

        # ---------------------------------------上面暂时未用到------------------------------------

        # multi grid dilation option
        parser.add_argument("--dilated", default=True, type=bool,
                            help="use dilation policy")
        parser.add_argument("--multi_grid", default=True, type=bool,
                            help="use multi grid dilation policy")
        parser.add_argument('--multi_dilation', type=int, default=[4, 8, 16],
                            help="multi grid dilation list")
        parser.add_argument('--scale', action='store_false', default=False,
                            help='choose to use random scale transform(0.75-2),default:multi scale')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs, batch_size and lr
        if args.epoch_num is None:
            epoches = {
                'pascal_voc': 50,
                'pascal_aug': 50,
                'pcontext': 80,
                'ade20k': 180,
                'cityscapes': 240,
            }
            num_class_dict = {
                'pascal_voc': 21,
                'pascal_aug': 21,
                'pcontext': 21,
                'ade20k': 10,
                'cityscapes': 19,
            }
            total_steps = {
                'pascal_voc': 90000,
                'pascal_aug': 90000,
                'pcontext': 90000,
                'ade20k': 90000,
                'cityscapes': 90000,
            }
            args.epoch_num = epoches[args.dataset.lower()]
            args.num_classes = num_class_dict[args.dataset.lower()]
            args.total_step = total_steps[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 5
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.step_per_epoch is None:
            step_per_epoch = {
                'pascal_voc': 185,
                'pascal_aug': 185,
                'pcontext': 185,
                'ade20k': 185,
                'cityscapes': 1,  # 2975 // batch_size // cuda_num
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


if __name__ == '__main__':
    opt = Options()
    args = opt.parse()
    print(args)
    opt.print_args()
