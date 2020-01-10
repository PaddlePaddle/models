# Copyright (c) 2018-present, Baidu, Inc.
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
##############################################################################
"""Functions for training."""

import os
import sys
import numpy as np
import cv2
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import argparse
import functools

from lib import pose_resnet
from utils.utility import *

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,   128,                          "Minibatch size totally.")
add_arg('dataset',          str,   'mpii',                       "Dataset, valid value: mpii, coco")
add_arg('use_gpu',          bool,  True,                         "Whether to use GPU or not.")
add_arg('num_epochs',       int,   140,                          "Number of epochs.")
add_arg('total_images',     int,   144406,                       "Training image number.")
add_arg('kp_dim',           int,   16,                           "Class number.")
add_arg('model_save_dir',   str,   "output",                     "Model save directory")
add_arg('pretrained_model', str,   "pretrained/resnet_50/115",   "Whether to use pretrained model.")
add_arg('checkpoint',       str,   None,                         "Whether to resume checkpoint.")
add_arg('lr',               float, 0.001,                        "Set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",            "Set the learning rate decay strategy.")
add_arg('enable_ce',        bool,  False,                        "If set True, enable continuous evaluation job.")
# yapf: enable


def optimizer_setting(args, params):
    lr_drop_ratio = 0.1

    ls = params["learning_strategy"]

    if ls["name"] == "piecewise_decay":
        total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        ls['epochs'] = [90, 120]
        print('=> LR will be dropped at the epoch of {}'.format(ls['epochs']))

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (lr_drop_ratio**i) for i in range(len(bd) + 1)]

        # AdamOptimizer
        optimizer = paddle.fluid.optimizer.AdamOptimizer(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr))
    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(0.0005))

    return optimizer


def print_immediately(s):
    print(s)
    sys.stdout.flush()


def train(args):
    if args.dataset == 'coco':
        import lib.coco_reader as reader
        IMAGE_SIZE = [288, 384]
        HEATMAP_SIZE = [72, 96]
        args.kp_dim = 17
        args.total_images = 144406  # 149813
    elif args.dataset == 'mpii':
        import lib.mpii_reader as reader
        IMAGE_SIZE = [384, 384]
        HEATMAP_SIZE = [96, 96]
        args.kp_dim = 16
        args.total_images = 22246
    else:
        raise ValueError('The dataset {} is not supported yet.'.format(
            args.dataset))

    print_arguments(args)

    # Image and target
    image = layers.data(
        name='image', shape=[3, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype='float32')
    target = layers.data(
        name='target',
        shape=[args.kp_dim, HEATMAP_SIZE[1], HEATMAP_SIZE[0]],
        dtype='float32')
    target_weight = layers.data(
        name='target_weight', shape=[args.kp_dim, 1], dtype='float32')

    # used for ce
    if args.enable_ce:
        fluid.default_startup_program().random_seed = 90
        fluid.default_main_program().random_seed = 90

    # Build model
    model = pose_resnet.ResNet(layers=50, kps_num=args.kp_dim)

    # Output
    loss, output = model.net(input=image,
                             target=target,
                             target_weight=target_weight)

    # Parameters from model and arguments
    params = {}
    params["total_images"] = args.total_images
    params["lr"] = args.lr
    params["num_epochs"] = args.num_epochs
    params["learning_strategy"] = {}
    params["learning_strategy"]["batch_size"] = args.batch_size
    params["learning_strategy"]["name"] = args.lr_strategy

    # Initialize optimizer
    optimizer = optimizer_setting(args, params)
    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:

        def if_exist(var):
            exist_flag = os.path.exists(
                os.path.join(args.pretrained_model, var.name))
            return exist_flag

        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    if args.checkpoint is not None:
        fluid.io.load_persistables(exe, args.checkpoint)

    # Dataloader
    train_reader = paddle.batch(reader.train(), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, target, target_weight])

    train_exe = fluid.ParallelExecutor(
        use_cuda=True if args.use_gpu else False, loss_name=loss.name)
    fetch_list = [image.name, loss.name, output.name]

    for pass_id in range(params["num_epochs"]):
        for batch_id, data in enumerate(train_reader()):
            current_lr = np.array(paddle.fluid.global_scope().find_var(
                'learning_rate').get_tensor())

            input_image, loss, out_heatmaps = train_exe.run(
                fetch_list, feed=feeder.feed(data))

            loss = np.mean(np.array(loss))

            print_immediately('Epoch [{:4d}/{:3d}] LR: {:.10f} '
                              'Loss = {:.5f}'.format(batch_id, pass_id,
                                                     current_lr[0], loss))

            if batch_id % 10 == 0:
                save_batch_heatmaps(
                    input_image,
                    out_heatmaps,
                    file_name='visualization@train.jpg',
                    normalize=True)

        model_path = os.path.join(
            args.model_save_dir + '/' + 'simplebase-{}'.format(args.dataset),
            str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)

    # used for ce
    if args.enable_ce:
        device_num = fluid.core.get_cuda_device_count() if args.use_gpu else 1
        print("kpis\t{}_train_cost_card{}\t{:.5f}".format(args.dataset,
                                                          device_num, loss))


if __name__ == '__main__':
    args = parser.parse_args()
    check_cuda(args.use_gpu)
    train(args)
