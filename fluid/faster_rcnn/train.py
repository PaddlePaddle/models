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
from fasterrcnn_model import FasterRcnn, RPNloss
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
# SOLVER
add_arg('learning_rate',    float,  0.01,     "Learning rate.")
add_arg('num_passes',       int,    20,        "Epoch number.")
# RPN
add_arg('anchor_sizes',     int,    [32,64,128,256,512],  "The size of anchors.")
add_arg('aspect_ratios',    float,  [0.5,1.0,2.0],    "The ratio of anchors.")
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

def load(file_name):
    v = cPickle.load(open(file_name))
    return v.astype('float32')

def train(args):
    num_passes = args.num_passes
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    image_shape = [3, args.max_size, args.max_size]


    if args.debug:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        import random
        random.seed(0)
        np.random.seed(0)


    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1],  dtype='int32', lod_level=1)
    is_crowd = fluid.layers.data(
        name='is_crowd', shape = [-1], dtype='int32', lod_level=1, append_batch_size=False)
    im_info = fluid.layers.data(
        name='im_info', shape=[3], dtype='float32')


    rpn_cls_score, rpn_bbox_pred, anchor, var, cls_score, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois, labels_int32 = FasterRcnn(
            input=image,
            depth=50,
            anchor_sizes=[32,64,128,256,512],
            variance=[1.,1.,1.,1.],
            aspect_ratios=[0.5,1.0,2.0],
            gt_box=gt_box,
            is_crowd=is_crowd,
            gt_label=gt_label,
            im_info=im_info,
            class_nums=args.class_nums,
            use_random=False if args.debug else True
            )

    cls_loss, reg_loss = RPNloss(rpn_cls_score, rpn_bbox_pred, anchor, var, gt_box, is_crowd, im_info, use_random=False if args.debug else True)
    cls_loss.persistable=True
    reg_loss.persistable=True
    rpn_loss = cls_loss + reg_loss
    rpn_loss.persistable=True

    labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
    labels_int64.stop_gradient = True
    #loss_cls = fluid.layers.softmax_with_cross_entropy(
    #        logits=cls_score,
    #        label=labels_int64
    #        )
    softmax = fluid.layers.softmax(cls_score, use_cudnn=False)
    loss_cls = fluid.layers.cross_entropy(softmax, labels_int64)
    loss_cls = fluid.layers.reduce_mean(loss_cls)
    loss_cls.persistable=True
    loss_bbox = fluid.layers.smooth_l1(x=bbox_pred,
                                        y=bbox_targets,
                                        inside_weight=bbox_inside_weights,
                                        outside_weight=bbox_outside_weights,
                                        sigma=1.0)
    loss_bbox = fluid.layers.reduce_mean(loss_bbox)
    loss_bbox.persistable=True

    loss_cls.persistable=True
    loss_bbox.persistable=True
    detection_loss = loss_cls + loss_bbox
    detection_loss.persistable=True

    loss = rpn_loss + detection_loss
    loss.persistable=True

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

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))
        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)


    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(args.use_gpu), loss_name=loss.name)

    train_reader = reader.train(args)

    def save_model(postfix):
        model_path = os.path.join(args.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(exe, model_path)

    fetch_list = [loss, cls_loss, reg_loss, loss_cls, loss_bbox]

    def tensor(data, place, lod=None):
        t = fluid.core.LoDTensor()
        t.set(data, place)
        if lod:
            t.set_lod(lod)
        return t

    total_time = 0.0
    for epoc_id in range(num_passes):
        start_time = time.time()
        prev_start_time = start_time
        every_pass_loss = []
        iter = 0
        pass_duration = 0.0
        for batch_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()

            image, gt_box, gt_label, is_crowd, im_info, lod = data
            image_t = tensor(image, place)
            gt_box_t = tensor(gt_box, place, [lod])
            gt_label_t = tensor(gt_label, place, [lod])
            is_crowd_t = tensor(is_crowd, place, [lod])
            im_info_t = tensor(im_info, place)

            feeding = {}
            feeding['image'] = image_t
            feeding['gt_box'] = gt_box_t
            feeding['gt_label'] = gt_label_t
            feeding['is_crowd'] = is_crowd_t
            feeding['im_info'] = im_info_t

            if args.parallel:
                losses = train_exe.run(fetch_list=[v.name for v in fetch_list],
                                       feed=feeding)
            else:
                losses = exe.run(fluid.default_main_program(),
                                  feed=feeding,
                                  fetch_list=fetch_list)
            loss_v = np.mean(np.array(losses[0]))
            every_pass_loss.append(loss_v)

            lr = np.array(fluid.global_scope().find_var('learning_rate').get_tensor())
            if batch_id % 1 == 0:
                print("Epoc {:d}, batch {:d}, lr {:.6f}, loss {:.6f}, time {:.5f}".format(
                      epoc_id, batch_id, lr[0], losses[0][0], start_time - prev_start_time))
                #print('cls_loss ', losses[1][0], ' reg_loss ', losses[2][0], ' loss_cls ', losses[3][0], ' loss_bbox ', losses[4][0])

        if epoc_id % 10 == 0 or epoc_id == num_passes - 1:
            save_model(str(epoc_id))

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_args = reader.Settings(args)
    train(data_args)
