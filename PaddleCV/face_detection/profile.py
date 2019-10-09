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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
import time
import argparse
import functools

import reader
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from pyramidbox import PyramidBox
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# yapf: disable
add_arg('parallel',         bool,  True,            "parallel")
add_arg('learning_rate',    float, 0.001,           "Learning rate.")
add_arg('batch_size',       int,   20,              "Minibatch size.")
add_arg('num_iteration',    int,   10,              "Epoch number.")
add_arg('use_gpu',          bool,  True,            "Whether use GPU.")
add_arg('use_pyramidbox',   bool,  True,            "Whether use PyramidBox model.")
add_arg('model_save_dir',   str,   'output',        "The path to save model.")
add_arg('pretrained_model', str,   './vgg_ilsvrc_16_fc_reduced', "The init model path.")
add_arg('resize_h',         int,   640,             "The resized image height.")
add_arg('resize_w',         int,   640,             "The resized image height.")
add_arg('data_dir',         str,   'data',          "The base dir of dataset")
#yapf: enable


def train(args, config, train_file_list, optimizer_method):
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    height = args.resize_h
    width = args.resize_w
    use_gpu = args.use_gpu
    use_pyramidbox = args.use_pyramidbox
    model_save_dir = args.model_save_dir
    pretrained_model = args.pretrained_model
    num_iterations = args.num_iteration
    parallel = args.parallel

    num_classes = 2
    image_shape = [3, height, width]

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=8,
            shapes=[[-1] + image_shape, [-1, 4], [-1, 4], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "float32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, face_box, head_box, gt_label = fluid.layers.read_file(py_reader)
            fetches = []
            network = PyramidBox(image=image,
                                 face_box=face_box,
                                 head_box=head_box,
                                 gt_label=gt_label,
                                 sub_network=use_pyramidbox)
            if use_pyramidbox:
                face_loss, head_loss, loss = network.train()
                fetches = [face_loss, head_loss]
            else:
                loss = network.vgg_ssd_loss()
                fetches = [loss]
            devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
            devices_num = len(devices.split(","))
            batch_size_per_device = batch_size // devices_num
            steps_per_pass = 12880 // batch_size
            boundaries = [steps_per_pass * 50, steps_per_pass * 80,
                          steps_per_pass * 120, steps_per_pass * 140]
            values = [
                learning_rate, learning_rate * 0.5, learning_rate * 0.25,
                learning_rate * 0.1, learning_rate * 0.01]
            if optimizer_method == "momentum":
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=fluid.layers.piecewise_decay(
                        boundaries=boundaries, values=values),
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(0.0005),
                )
            else:
                optimizer = fluid.optimizer.RMSProp(
                    learning_rate=
                    fluid.layers.piecewise_decay(boundaries, values),
                    regularization=fluid.regularizer.L2Decay(0.0005),
                )
            optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    start_pass = 0
    if pretrained_model:
        if pretrained_model.isdigit():
            start_pass = int(pretrained_model) + 1
            pretrained_model = os.path.join(model_save_dir, pretrained_model)
            print("Resume from %s " %(pretrained_model))

        if not os.path.exists(pretrained_model):
            raise ValueError("The pre-trained model path [%s] does not exist." %
                             (pretrained_model))
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    if parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_gpu, loss_name=loss.name, main_program = train_prog)
    train_reader = reader.train(config,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=False,
                                use_multiprocessing=True,
                                num_workers=8,
                                max_queue=24)
    py_reader.decorate_paddle_reader(train_reader)

    def run(iterations):
        # global feed_data
        py_reader.start()
        run_time = []
        for batch_id in range(iterations):
            start_time = time.time()
            if parallel:
                fetch_vars = train_exe.run(fetch_list=[v.name for v in fetches])
            else:
                fetch_vars = exe.run(train_prog,
                                     fetch_list=fetches)
            end_time = time.time()
            run_time.append(end_time - start_time)
            fetch_vars = [np.mean(np.array(v)) for v in fetch_vars]
            if not args.use_pyramidbox:
                print("Batch {0}, loss {1}".format(batch_id, fetch_vars[0]))
            else:
                print("Batch {0}, face loss {1}, head loss {2}".format(
                       batch_id, fetch_vars[0], fetch_vars[1]))
        return run_time

    # start-up
    run(2)

    # profiling
    start = time.time()
    if not parallel:
        with profiler.profiler('All', 'total', '/tmp/profile_file'):
            run_time = run(num_iterations)
    else:
        run_time = run(num_iterations)
    end = time.time()
    total_time = end - start
    print("Total time: {0}, reader time: {1} s, run time: {2} s".format(
        total_time, total_time - np.sum(run_time), np.sum(run_time)))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = os.path.join(args.data_dir, 'WIDER_train/images/')
    train_file_list = os.path.join(args.data_dir,
        'wider_face_split/wider_face_train_bbx_gt.txt')

    config = reader.Settings(
        data_dir=data_dir,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        apply_expand=False,
        mean_value=[104., 117., 123.],
        ap_version='11point')
    train(args, config, train_file_list, optimizer_method="momentum")
