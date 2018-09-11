import os
import time
import numpy as np
import argparse
import functools
import shutil
import paddle
import paddle.fluid as fluid
import reader
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
from paddle.fluid.layers import control_flow
from fasterrcnn_model import FasterRcnn, RPNloss
from utility import add_arguments, print_arguments
from paddle.fluid.layers import tensor

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
add_arg('num_passes',       int,    100,        "Epoch number.")
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

def exponential_with_warmup_decay(learning_rate, boundaries, values, warmup_iter, warmup_factor):
    global_step = lr_scheduler._decay_step_counter()

    lr = fluid.layers.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    warmup_iter_var = fluid.layers.fill_constant(
        shape=[1],
        dtype='float32',
        value=float(warmup_iter),
        force_cpu=True)

    with control_flow.Switch() as switch:
        with switch.case(global_step < warmup_iter_var):
            alpha = global_step / warmup_iter_var
            factor = warmup_factor * (1 - alpha) + alpha
            decayed_lr = learning_rate * factor
            fluid.layers.assign(decayed_lr, lr)

        for i in range(len(boundaries)):
            boundary_val = fluid.layers.fill_constant(
                shape=[1],
                dtype='float32',
                value=float(boundaries[i]),
                force_cpu=True)
            value_var = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=float(values[i]))
            with switch.case(global_step < boundary_val):
                fluid.layers.assign(value_var, lr)

        last_value_var = fluid.layers.fill_constant(
            shape=[1], dtype='float32', value=float(values[len(values) - 1]))
        with switch.default():
            fluid.layers.assign(last_value_var, lr)

    return lr


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
            anchor_sizes=args.anchor_sizes,
            variance=[0.1,0.1,0.2,0.2],
            aspect_ratios=args.aspect_ratios,
            gt_box=gt_box,
            is_crowd=is_crowd,
            gt_label=gt_label,
            im_info=im_info,
            class_nums=args.class_nums,
            use_random=False if args.debug else True
            )

    cls_loss, reg_loss = RPNloss(rpn_cls_score, rpn_bbox_pred, anchor, var, gt_box, is_crowd, im_info, use_random=False if args.debug else True)
    rpn_loss = cls_loss + reg_loss
    rpn_loss.persistable=True

    labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
    labels_int64.stop_gradient = True
    loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=cls_score,
            label=labels_int64
            )
    loss_cls = fluid.layers.reduce_mean(loss_cls)
    loss_cls.persistable=True
    loss_bbox = fluid.layers.smooth_l1(x=bbox_pred,
                                        y=bbox_targets,
                                        inside_weight=bbox_inside_weights,
                                        outside_weight=bbox_outside_weights,
                                        sigma=1.0)
    loss_bbox = fluid.layers.reduce_mean(loss_bbox)
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

    if args.use_gpu:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:
        def if_exist(var):
            #if (os.path.exists(os.path.join(pretrained_model, var.name))):
            #   print(var.name)
            return os.path.exists(os.path.join(args.pretrained_model, var.name))
        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)


    if args.debug:
        import load_var
        rpn_bbox_pred_w_t, rpn_bbox_pred_b_t = load_var.load_rpn_bbox_pred()
        rpn_cls_logits_w_t, rpn_cls_logits_b_t = load_var.load_rpn_cls_logits()
        rpn_conv_w_t, rpn_conv_b_t = load_var.load_rpn_conv()
        cls_score_w_t, cls_score_b_t = load_var.load_cls_score()
        bbox_pred_w_t, bbox_pred_b_t = load_var.load_bbox_pred()

        conv_rpn_w = fluid.global_scope().find_var("conv_rpn_w").get_tensor()
        conv_rpn_b = fluid.global_scope().find_var("conv_rpn_b").get_tensor()
        conv_rpn_w.set(rpn_conv_w_t, place)
        conv_rpn_b.set(rpn_conv_b_t, place)

        rpn_cls_logits_w = fluid.global_scope().find_var("rpn_cls_logits_w").get_tensor()
        rpn_cls_logits_b = fluid.global_scope().find_var("rpn_cls_logits_b").get_tensor()
        rpn_cls_logits_w.set(rpn_cls_logits_w_t, place)
        rpn_cls_logits_b.set(rpn_cls_logits_b_t, place)

        rpn_bbox_pred_w = fluid.global_scope().find_var("rpn_bbox_pred_w").get_tensor()
        rpn_bbox_pred_b = fluid.global_scope().find_var("rpn_bbox_pred_b").get_tensor()
        rpn_bbox_pred_w.set(rpn_bbox_pred_w_t, place)
        rpn_bbox_pred_b.set(rpn_bbox_pred_b_t, place)

        cls_score_w = fluid.global_scope().find_var("cls_score_w").get_tensor()
        cls_score_b = fluid.global_scope().find_var("cls_score_b").get_tensor()
        cls_score_w.set(cls_score_w_t, place)
        cls_score_b.set(cls_score_b_t, place)

        bbox_pred_w = fluid.global_scope().find_var("bbox_pred_w").get_tensor()
        bbox_pred_b = fluid.global_scope().find_var("bbox_pred_b").get_tensor()
        bbox_pred_w.set(bbox_pred_w_t, place)
        bbox_pred_b.set(bbox_pred_b_t, place)

    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(args.use_gpu), loss_name=loss.name)
    """
    train_reader = paddle.batch(
        reader.train(args, shuffle=False if args.debug else True), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(args), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, is_crowd, im_info])
    """
    train_reader = reader.train(args)
    def save_model(postfix):
        model_path = os.path.join(args.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print 'save models to %s' % (model_path)
        fluid.io.save_persistables(exe, model_path)


    fetch_list = [loss, rpn_loss, detection_loss]

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
            lod = [lod]
            image_t = fluid.core.LoDTensor()
            image_t.set(image, place)

            gt_box_t = fluid.core.LoDTensor()
            gt_box_t.set(gt_box, place)
            gt_box_t.set_lod(lod)

            gt_label_t = fluid.core.LoDTensor()
            gt_label_t.set(gt_label.reshape(-1, 1), place)
            gt_label_t.set_lod(lod)

            is_crowd_t = fluid.core.LoDTensor()
            is_crowd_t.set(is_crowd, place)
            is_crowd_t.set_lod(lod)

            im_info_t = fluid.core.LoDTensor()
            im_info_t.set(im_info, place)

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
            lr = np.array(fluid.global_scope().find_var('learning_rate').get_tensor())

            loss_v = np.mean(np.array(losses[0]))
            every_pass_loss.append(loss_v)
            if batch_id % 1 == 0:
                print("Epoc {:d}, batch {:d}, lr {:.6f}, loss {:.6f}, time {:.5f}".format(
                      epoc_id, batch_id, lr[0], losses[0][0], start_time - prev_start_time))

        if epoc_id % 10 == 0 or epoc_id == num_passes - 1:
            save_model(str(epoc_id))

if __name__ == '__main__':
    print('fasterrcnn')
    args = parser.parse_args()
    print_arguments(args)

    data_args = reader.Settings(args)
    print('train begins...')
    train(data_args)
