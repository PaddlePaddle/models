# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#order: standard library, third party, local library 
import os
import time
import sys
import math
import argparse
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid import framework
import reader
from utils import *
from mobilenet_v1 import *
from mobilenet_v2 import *

from imagenet_dataset import ImageNetDataset
from paddle.io import DataLoader

args = parse_args()
if int(os.getenv("PADDLE_TRAINER_ID", 0)) == 0:
    print_arguments(args)


class TimeCostAverage(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt = 0
        self.total_time = 0

    def record(self, usetime):
        self.cnt += 1
        self.total_time += usetime

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.total_time / self.cnt


def eval(net, test_data_loader, eop):
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    total_sample = 0
    t_last = 0
    place_num = paddle.fluid.core.get_cuda_device_count(
    ) if args.use_gpu else int(os.environ.get('CPU_NUM', 1))

    batch_start = time.time()
    for img, label in test_data_loader():
        batch_reader_end = time.time()

        label = to_variable(label.numpy().astype('int64').reshape(
            int(args.batch_size // place_num), 1))
        out = net(img)
        softmax_out = fluid.layers.softmax(out, use_cudnn=False)
        loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
        avg_loss = fluid.layers.mean(x=loss)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
        total_loss += avg_loss.numpy()
        total_acc1 += acc_top1.numpy()
        total_acc5 += acc_top5.numpy()

        test_batch_cost = time.time() - batch_start
        total_sample += 1
        print(
            "test | epoch %d, avg_loss %.5f, acc_top1 %.5f, acc_top5 %.5f, batch_cost_time: %.5f s, reader_cost: %.5f s"
            % (eop, avg_loss.numpy(), acc_top1.numpy(), acc_top5.numpy(),
               test_batch_cost, batch_reader_end - batch_start))
        sys.stdout.flush()
        batch_start = time.time()

    print("final eval loss %0.3f acc1 %0.3f acc5 %0.3f" % \
          (total_loss / total_sample, \
           total_acc1 / total_sample, total_acc5 / total_sample))
    sys.stdout.flush()


def train_mobilenet():
    if not args.use_gpu:
        place = fluid.CPUPlace()
    elif not args.use_data_parallel:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):
        # 1. init net and optimizer
        place_num = paddle.fluid.core.get_cuda_device_count(
        ) if args.use_gpu else int(os.environ.get('CPU_NUM', 1))
        if args.ce:
            print("ce mode")
            seed = 33
            np.random.seed(seed)
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
        if args.use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        if args.model == "MobileNetV1":
            net = MobileNetV1(class_dim=args.class_dim, scale=1.0)
            model_path_pre = 'mobilenet_v1'
        elif args.model == "MobileNetV2":
            net = MobileNetV2(class_dim=args.class_dim, scale=1.0)
            model_path_pre = 'mobilenet_v2'
        else:
            print(
                "wrong model name, please try model = MobileNetV1 or MobileNetV2"
            )
            exit()

        optimizer = create_optimizer(args=args, parameter_list=net.parameters())
        if args.use_data_parallel:
            net = fluid.dygraph.parallel.DataParallel(net, strategy)

        # 2. load checkpoint
        if args.checkpoint:
            assert os.path.exists(args.checkpoint + ".pdparams"), \
                "Given dir {}.pdparams not exist.".format(args.checkpoint)
            assert os.path.exists(args.checkpoint + ".pdopt"), \
                "Given dir {}.pdopt not exist.".format(args.checkpoint)
            para_dict, opti_dict = fluid.dygraph.load_dygraph(args.checkpoint)
            net.set_dict(para_dict)
            optimizer.set_dict(opti_dict)

        # 3. reader
        test_data_loader = utility.create_data_loader(is_train=False, args=args)
        num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
        imagenet_reader = reader.ImageNetReader(seed=0, place_num=place_num)

        train_dataset = ImageNetDataset(
            os.path.join(args.data_dir, "train"), mode='train')

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            places=place,
            shuffle=True,
            drop_last=True,
            num_workers=10)

        test_dataset = ImageNetDataset(
            os.path.join(args.data_dir, "val"), mode='val')

        test_data_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            places=place,
            shuffle=True,
            drop_last=True,
            num_workers=1)

        # 4. train loop
        total_batch_num = 0  #this is for benchmark
        for eop in range(args.num_epochs):
            epoch_start = time.time()

            if num_trainers > 1:
                imagenet_reader.set_shuffle_seed(eop + (
                    args.random_seed if args.random_seed else 0))

            net.train()
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0
            batch_id = 0
            t_last = 0

            # 4.1 for each batch, call net() , backward(), and minimize()
            batch_cost_avg = TimeCostAverage()
            batch_reader_avg = TimeCostAverage()
            batch_net_avg = TimeCostAverage()
            batch_backward_avg = TimeCostAverage()
            batch_start = time.time()
            for img, label in train_data_loader():
                if args.max_iter and total_batch_num == args.max_iter:
                    return
                batch_reader_end = time.time()

                # 4.1.1 call net()
                out = net(img)
                softmax_out = fluid.layers.softmax(out, use_cudnn=False)
                loss = fluid.layers.cross_entropy(
                    input=softmax_out, label=label)
                avg_loss = fluid.layers.mean(x=loss)
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
                batch_net_end = time.time()

                # 4.1.2 call backward()
                if args.use_data_parallel:
                    avg_loss = net.scale_loss(avg_loss)
                    avg_loss.backward()
                    net.apply_collective_grads()
                else:
                    avg_loss.backward()
                batch_backward_end = time.time()

                # 4.1.3 call minimize()
                optimizer.minimize(avg_loss)

                net.clear_gradients()
                t2 = time.time()

                avg_loss_value = avg_loss.numpy()
                acc_top1_value = acc_top1.numpy()
                acc_top5_value = acc_top5.numpy()

                total_loss += avg_loss_value
                total_acc1 += acc_top1_value
                total_acc5 += acc_top5_value

                total_sample += 1
                batch_id += 1

                # NOTE: used for benchmark
                train_batch_cost = time.time() - batch_start
                batch_cost_avg.record(train_batch_cost)
                batch_reader_avg.record(batch_reader_end - batch_start)
                batch_net_avg.record(batch_net_end - batch_reader_end)
                batch_backward_avg.record(batch_backward_end - batch_net_end)

                total_batch_num = total_batch_num + 1
                if batch_id % args.print_step == 0:
                    ips = float(args.batch_size) / batch_cost_avg.get_average()
                    print(
                        "[Epoch %d, batch %d], avg_loss %.5f, acc_top1 %.5f, acc_top5 %.5f, batch_cost: %.5f sec, net_cost: %.5f sec, backward_cost: %.5f sec, reader_cost: %.5f sec, ips: %.5f images/sec"
                        % (eop, batch_id, avg_loss_value, acc_top1_value,
                           acc_top5_value, batch_cost_avg.get_average(),
                           batch_net_avg.get_average(),
                           batch_backward_avg.get_average(),
                           batch_reader_avg.get_average(), ips))
                    sys.stdout.flush()
                    batch_cost_avg.reset()
                    batch_net_avg.reset()
                    batch_backward_avg.reset()
                    batch_reader_avg.reset()
                batch_start = time.time()

            if args.ce:
                print("kpis\ttrain_acc1\t%0.3f" % (total_acc1 / total_sample))
                print("kpis\ttrain_acc5\t%0.3f" % (total_acc5 / total_sample))
                print("kpis\ttrain_loss\t%0.3f" % (total_loss / total_sample))

            train_epoch_cost = time.time() - epoch_start
            print(
                "[Epoch %d], loss %.5f, acc1 %.5f, acc5 %.5f, epoch_cost: %.5f s"
                % (eop, total_loss / total_sample, total_acc1 / total_sample,
                   total_acc5 / total_sample, train_epoch_cost))

            # 4.2 save checkpoint
            save_parameters = (not args.use_data_parallel) or (
                args.use_data_parallel and
                fluid.dygraph.parallel.Env().local_rank == 0)
            if save_parameters:
                if not os.path.isdir(args.model_save_dir):
                    os.makedirs(args.model_save_dir)
                model_path = os.path.join(
                    args.model_save_dir,
                    "_" + model_path_pre + "_epoch{}".format(eop))
                fluid.dygraph.save_dygraph(net.state_dict(), model_path)
                fluid.dygraph.save_dygraph(optimizer.state_dict(), model_path)

            # 4.3 validation
            net.eval()
            eval(net, test_data_loader, eop)

        # 5. save final results
        save_parameters = (not args.use_data_parallel) or (
            args.use_data_parallel and
            fluid.dygraph.parallel.Env().local_rank == 0)
        if save_parameters:
            model_path = os.path.join(args.model_save_dir,
                                      "_" + model_path_pre + "_final")
            fluid.dygraph.save_dygraph(net.state_dict(), model_path)


if __name__ == '__main__':
    train_mobilenet()
