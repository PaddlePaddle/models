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
import time
import numpy as np
import argparse
import functools
import shutil
import math
import multiprocessing
from paddle.fluid.contrib.slim import Compressor


def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect. 
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import build_mobilenet_ssd
from utility import add_arguments, print_arguments, check_cuda

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.001,     "Learning rate.")
add_arg('batch_size',       int,   64,        "Minibatch size of all devices.")
add_arg('epoc_num',         int,   120,       "Epoch number.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('parallel',         bool,  True,      "Whether train in parallel on multi-devices.")
add_arg('dataset',          str,   'pascalvoc', "dataset can be coco2014, coco2017, and pascalvoc.")
add_arg('model_save_dir',   str,   'model',     "The path to save model.")
add_arg('pretrained_model', str,   'pretrained/ssd_mobilenet_v1_coco/', "The init model path.")
add_arg('ap_version',       str,   '11point',           "mAP version can be integral or 11point.")
add_arg('image_shape',      str,   '3,300,300',         "Input image shape.")
add_arg('mean_BGR',         str,   '127.5,127.5,127.5', "Mean value for B,G,R channel which will be subtracted.")
add_arg('data_dir',         str,   'data/pascalvoc', "Data directory.")
add_arg('use_multiprocess', bool,  True,  "Whether use multi-process for data preprocessing.")
add_arg('enable_ce',        bool,  False, "Whether use CE to evaluate the model.")
#yapf: enable

train_parameters = {
    "pascalvoc": {
        "train_images": 16551,
        "image_shape": [3, 300, 300],
        "class_num": 21,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01],
        "ap_version": '11point',
    },
    "coco2014": {
        "train_images": 82783,
        "image_shape": [3, 300, 300],
        "class_num": 91,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [12, 19],
        "lr_decay": [1, 0.5, 0.25],
        "ap_version": 'integral', # should use eval_coco_map.py to test model
    },
    "coco2017": {
        "train_images": 118287,
        "image_shape": [3, 300, 300],
        "class_num": 91,
        "batch_size": 64,
        "lr": 0.001,
        "lr_epochs": [12, 19],
        "lr_decay": [1, 0.5, 0.25],
        "ap_version": 'integral', # should use eval_coco_map.py to test model
    }
}

def optimizer_setting(train_params):
    batch_size = train_params["batch_size"]
    iters = train_params["train_images"] // batch_size
    lr = train_params["lr"]
    boundaries = [i * iters  for i in train_params["lr_epochs"]]
    values = [ i * lr for i in train_params["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    return optimizer


def build_program(main_prog, startup_prog, train_params, is_train):
    image_shape = train_params['image_shape']
    class_num = train_params['class_num']
    ap_version = train_params['ap_version']
    outs = []
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=64,
            shapes=[[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "int32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, gt_box, gt_label, difficult = fluid.layers.read_file(py_reader)
            locs, confs, box, box_var = build_mobilenet_ssd(image, class_num, image_shape)
            if is_train:
                with fluid.unique_name.guard("train"):
                    loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box,
                        box_var)
                    loss = fluid.layers.reduce_sum(loss)
                    optimizer = optimizer_setting(train_params)
                    optimizer.minimize(loss)
                outs = [py_reader, loss]
            else:
                with fluid.unique_name.guard("inference"):
                    nmsed_out = fluid.layers.detection_output(
                        locs, confs, box, box_var, nms_threshold=0.45)

                    gt_label = fluid.layers.cast(x=gt_label, dtype=gt_box.dtype)
                    if difficult:
                        difficult = fluid.layers.cast(x=difficult, dtype=gt_box.dtype)
                        gt_label = fluid.layers.reshape(gt_label, [-1, 1])
                        difficult = fluid.layers.reshape(difficult, [-1, 1])
                        label = fluid.layers.concat([gt_label, difficult, gt_box], axis=1)
                    else:
                        label = fluid.layers.concat([gt_label, gt_box], axis=1)

                    map_var = fluid.layers.detection.detection_map(
                            nmsed_out,
                            label,
                            class_num,
                            background_label=0,
                            overlap_threshold=0.5,
                            evaluate_difficult=False,
                            ap_version=ap_version)

                # nmsed_out and image is used to save mode for inference
                outs = [py_reader, map_var, nmsed_out, image]
    return outs


def train(args,
          data_args,
          train_params,
          train_file_list,
          val_file_list):

    model_save_dir = args.model_save_dir
    pretrained_model = args.pretrained_model
    use_gpu = args.use_gpu
    parallel = args.parallel
    enable_ce = args.enable_ce
    is_shuffle = True

    if not use_gpu:
        devices_num = int(os.environ.get('CPU_NUM',
                          multiprocessing.cpu_count()))
    else:
        devices_num = fluid.core.get_cuda_device_count()

    batch_size = train_params['batch_size']
    epoc_num = train_params['epoc_num']
    batch_size_per_device = batch_size // devices_num
    num_workers = 8

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()


    train_py_reader, loss = build_program(
        main_prog=train_prog,
        startup_prog=startup_prog,
        train_params=train_params,
        is_train=True)
    test_py_reader, map_var, _, _ = build_program(
        main_prog=test_prog,
        startup_prog=startup_prog,
        train_params=train_params,
        is_train=False)

    test_prog = test_prog.clone(for_test=True)
    
    for param in train_prog.global_block().all_parameters():
        if 'conv' in param.name:
            print param.name, param.shape
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, main_program=train_prog,
                           predicate=if_exist)


    test_reader = reader.test(data_args, val_file_list, batch_size)
    test_py_reader.decorate_paddle_reader(test_reader)
    train_reader = reader.train(data_args,
                                train_file_list,
                                batch_size_per_device,
                                shuffle=is_shuffle,
                                use_multiprocess=args.use_multiprocess,
                                num_workers=num_workers,
                                enable_ce=enable_ce)
    train_py_reader.decorate_paddle_reader(train_reader)

    train_fetch_list=[("loss", loss.name)]
    val_fetch_list=[("map",  map_var.name)]
    compressor = Compressor(
        place,
        fluid.global_scope(),
        train_prog,
        train_reader=train_py_reader,
        train_feed_list=None,
        train_fetch_list=train_fetch_list,
        eval_program=test_prog,
        eval_reader=test_py_reader,
        eval_feed_list=None,
        eval_fetch_list=val_fetch_list,
        train_optimizer=None)
    compressor.config('./compress.yaml')
    compressor.run()


def main():
    args = parser.parse_args()
    print_arguments(args)

    check_cuda(args.use_gpu)

    data_dir = args.data_dir
    dataset = args.dataset
    assert dataset in ['pascalvoc', 'coco2014', 'coco2017']

    # for pascalvoc
    label_file = 'label_list'
    train_file_list = 'trainval.txt'
    val_file_list = 'test.txt'

    if dataset == 'coco2014':
        train_file_list = 'annotations/instances_train2014.json'
        val_file_list = 'annotations/instances_val2014.json'
    elif dataset == 'coco2017':
        train_file_list = 'annotations/instances_train2017.json'
        val_file_list = 'annotations/instances_val2017.json'

    mean_BGR = [float(m) for m in args.mean_BGR.split(",")]
    image_shape = [int(m) for m in args.image_shape.split(",")]
    train_parameters[dataset]['image_shape'] = image_shape
    train_parameters[dataset]['batch_size'] = args.batch_size
    train_parameters[dataset]['lr'] = args.learning_rate
    train_parameters[dataset]['epoc_num'] = args.epoc_num
    train_parameters[dataset]['ap_version'] = args.ap_version

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        label_file=label_file,
        resize_h=image_shape[1],
        resize_w=image_shape[2],
        mean_value=mean_BGR,
        apply_distort=True,
        apply_expand=True,
        ap_version = args.ap_version)
    train(args,
          data_args,
          train_parameters[dataset],
          train_file_list=train_file_list,
          val_file_list=val_file_list)


if __name__ == '__main__':
    main()
