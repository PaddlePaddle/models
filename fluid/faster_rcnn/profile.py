import os
import time
import numpy as np
import argparse
import functools
import shutil
import cPickle
from utility import add_arguments, print_arguments

import paddle
import paddle.fluid as fluid
import reader
import paddle.fluid.profiler as profiler

import models.model_builder as model_builder
import models.resnet as resnet
from learning_rate import exponential_with_warmup_decay

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
# ENV
add_arg('parallel',         bool,   True,       "Minibatch size.")
add_arg('use_gpu',          bool,   True,      "Whether use GPU.")
add_arg('model_save_dir',   str,    'model',     "The path to save model.")
add_arg('pretrained_model', str,    'imagenet_resnet50_fusebn', "The init model path.")
add_arg('dataset',          str,    'coco2017', "coco2014, coco2017, and pascalvoc.")
add_arg('data_dir',         str,    'data/COCO17', "data directory")
add_arg('skip_reader',      bool,  False,            "Whether to skip data reader.")
add_arg('use_profile',      bool,  False,            "Whether to use profiler tool.")
add_arg('class_num',        int,   81,          "Class number.")
add_arg('use_pyreader',     bool,  False,          "Class number.")
# SOLVER
add_arg('learning_rate',    float,  0.01,     "Learning rate.")
add_arg('num_iteration',    int,   10,              "Epoch number.")
# RPN
add_arg('anchor_sizes',     int,    [32,64,128,256,512],  "The size of anchors.")
add_arg('aspect_ratios',    float,  [0.5,1.0,2.0],    "The ratio of anchors.")
add_arg('variance',         float,  [1.,1.,1.,1.],    "The variance of anchors.")
add_arg('rpn_stride',       float,  16.,    "Stride of the feature map that RPN is attached.")
# FAST RCNN
# TRAIN TEST
add_arg('batch_size',       int,    1,          "Minibatch size.")
add_arg('max_size',         int,    1333,    "The max resized image size.")
add_arg('scales',           int,    [800],    "The resized image height.")
add_arg('batch_size_per_im',int,    512,    "fast rcnn head batch size")
add_arg('mean_value',       float,  [102.9801, 115.9465, 122.7717], "pixel mean")
add_arg('debug',            bool,   False,   "Debug mode")
#yapf: enable

def train(cfg):
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    image_shape = [3, cfg.max_size, cfg.max_size]
    num_iterations = cfg.num_iteration

    if cfg.debug:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        import random
        random.seed(0)
        np.random.seed(0)

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    model = model_builder.FasterRCNN(
        cfg=cfg,
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=cfg.use_pyreader,
        use_random=False)
    model.build_model(image_shape)
    loss_cls, loss_bbox, rpn_cls_loss, rpn_reg_loss = model.loss()
    loss_cls.persistable=True
    loss_bbox.persistable=True
    rpn_cls_loss.persistable=True
    rpn_reg_loss.persistable=True
    loss = loss_cls + loss_bbox + rpn_cls_loss + rpn_reg_loss

    boundaries = [120000, 160000]
    values = [learning_rate, learning_rate*0.1, learning_rate*0.01]

    optimizer = fluid.optimizer.Momentum(
        learning_rate=exponential_with_warmup_decay(learning_rate=learning_rate,
            boundaries=boundaries,
            values=values,
            warmup_iter=500,
            warmup_factor=1.0/3.0),
        regularization=fluid.regularizer.L2Decay(0.0001),
        momentum=0.9)
    optimizer.minimize(loss)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)

    if cfg.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(cfg.use_gpu), loss_name=loss.name)


    if cfg.use_pyreader:
        train_reader = reader.train(cfg, batch_size=1, shuffle=not cfg.debug)
        py_reader = model.py_reader
        py_reader.decorate_paddle_reader(train_reader)
    else:
        train_reader = reader.train(cfg, batch_size=cfg.batch_size, shuffle=not cfg.debug)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    fetch_list = [loss, loss_cls, loss_bbox, rpn_cls_loss, rpn_reg_loss]

    def run(iterations):
        reader_time = []
        run_time = []
        total_images = 0

        for batch_id in range(iterations):
            start_time = time.time()
            data = train_reader().next()
            end_time = time.time()
            reader_time.append(end_time - start_time)
            start_time = time.time()
            losses = train_exe.run(fetch_list=[v.name for v in fetch_list],
                                   feed=feeder.feed(data))
            end_time = time.time()
            run_time.append(end_time - start_time)
            total_images += data[0][0].shape[0]

            lr = np.array(fluid.global_scope().find_var('learning_rate').get_tensor())
            print("Batch {:d}, lr {:.6f}, loss {:.6f} ".format(
                  batch_id, lr[0], losses[0][0]))
        return reader_time, run_time, total_images


    def run_pyreader(iterations):
        reader_time = [0]
        run_time = []
        total_images = 0

        py_reader.start()
        try:
            for batch_id in range(iterations):
                start_time = time.time()
                losses = train_exe.run(fetch_list=[v.name for v in fetch_list])
                end_time = time.time()
                run_time.append(end_time - start_time)
                total_images += devices_num
                lr = np.array(fluid.global_scope().find_var('learning_rate').get_tensor())
                print("Batch {:d}, lr {:.6f}, loss {:.6f} ".format(
                      batch_id, lr[0], losses[0][0]))
        except fluid.core.EOFException:
            py_reader.reset()

        return reader_time, run_time, total_images

    run_func = run if not cfg.use_pyreader else run_pyreader

    # warm-up
    run_func(2)
    # profiling
    start = time.time()
    if cfg.use_profile:
        with profiler.profiler('GPU', 'total', '/tmp/profile_file'):
            reader_time, run_time, total_images = run(num_iterations)
    else:
        reader_time, run_time, total_images = run_func(num_iterations)

    end = time.time()
    total_time = end - start
    print("Total time: {0}, reader time: {1} s, run time: {2} s, images/s: {3}".format(
        total_time, np.sum(reader_time), np.sum(run_time), total_images / total_time))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_args = reader.Settings(args)
    train(data_args)
