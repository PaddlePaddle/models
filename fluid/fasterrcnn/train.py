import os
import time
import numpy as np
import argparse
import functools
import shutil

import paddle
import paddle.fluid as fluid
import reader
from fasterrcnn_model import FasterRcnn, RPNloss
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.0001,     "Learning rate.")
add_arg('batch_size',       int,   8,        "Minibatch size.")
add_arg('num_passes',       int,   120,       "Epoch number.")
add_arg('use_gpu',          bool,  False,      "Whether use GPU.")
add_arg('parallel',         bool,  False,      "Parallel.")
add_arg('dataset',          str,   'pascalvoc', "coco2014, coco2017, and pascalvoc.")
add_arg('model_save_dir',   str,   'model',     "The path to save model.")
add_arg('pretrained_model', str,   None, "The init model path.")
add_arg('nms_threshold',    float, 0.5,   "NMS threshold.")
add_arg('anchor_sizes',      int,   [32,64,128],  "The size of anchors.")
add_arg('aspect_ratios',    float,   [0.5,1.0,2.0],    "The ratio of anchors.")
add_arg('ap_version',       str,   '11point',   "integral, 11point.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_R',     float, 127.77,  "Mean value for B channel which will be subtracted.")
add_arg('mean_value_G',     float, 115.95,  "Mean value for G channel which will be subtracted.")
add_arg('mean_value_B',     float, 102.98,  "Mean value for R channel which will be subtracted.")
add_arg('is_toy',           int,   0, "Toy for quick debug, 0 means using all data, while n means using only n sample.")
add_arg('data_dir',         str,   'data/pascalvoc', "data directory")
add_arg('enable_ce',     bool,  False, "Whether use CE to evaluate the model")
#yapf: enable


def train(args,
          train_file_list,
          val_file_list,
          data_args,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir,
          pretrained_model=None):
    if args.enable_ce:
        fluid.framework.default_startup_program().random_seed = 111

    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if 'coco' in data_args.dataset:
        class_nums = 81
    elif 'pascalvoc' in data_args.dataset:
        class_nums = 21

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1],  dtype='int32', lod_level=1)
    im_info = fluid.layers.data(
        name='im_info', shape=[3], dtype='float32')

    rpn_cls_score, rpn_bbox_pred, anchor, var, cls_score, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois, labels_int32 = FasterRcnn(
            input=image,
            depth=50,
            anchor_sizes=args.anchor_sizes,
            variance=[0.1,0.1,0.2,0.2],
            aspect_ratios=args.aspect_ratios,
            gt_box=gt_box,
            gt_label=gt_label,
            im_info=im_info,
            class_nums=class_nums,
            )

    cls_loss, reg_loss = RPNloss(rpn_cls_score, rpn_bbox_pred, anchor, var, gt_box)
    rpn_loss = cls_loss + reg_loss
    labels_int32 = fluid.layers.unsqueeze(input=labels_int32, axes=[1])
    loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=cls_score,
            label=labels_int32
            )
    loss_bbox = fluid.layers.smooth_l1(x=bbox_pred,
                                        y=bbox_targets,
                                        inside_weight=bbox_inside_weights,
                                        outside_weight=bbox_outside_weights,
                                        sigma=3.0)
    loss_cls = fluid.layers.reduce_sum(loss_cls)
    loss_bbox = fluid.layers.reduce_sum(loss_bbox)
    detection_loss = loss_cls + loss_bbox
    loss = rpn_loss + detection_loss

    """
    nmsed_rpn_out = fluid.layers.detection_output(rpn_bbox_pred, rpn_cls_score, 
                        anchor, var, nms_threshold=args.nms_threshold)
    nmsed_detection_out = fluid.layers.detection_output(bbox_pred, cls_score, 
                        rois, ???, nms_threshold=args.nms_threshold)

    test_program = fluid.default_main_program().clone(for_test=True)
    
    with fluid.program_guard(test_program):
        map_eval_rpn = fluid.evaluator.DetectionMAP(
            nmsed_rpn_out,
            gt_label,
            gt_box,
            num_classes,
            overlap_threshold=0.5,
            evaluate_difficult=False,
            ap_version=args.ap_version)
        map_eval_detection = fluid.evaluator.DetectionMAP(
            nmsed_detection_out,
            gt_label,
            gt_box,
            num_classes,
            overlap_threshold=0.5,
            evaluate_difficult=False,
            ap_version=args.ap_version)
    """
    if 'coco' in data_args.dataset:
        # learning rate decay in 12, 19 pass, respectively
        if '2014' in train_file_list:
            epocs = 82783 / batch_size
            boundaries = [epocs * 12, epocs * 19]
        elif '2017' in train_file_list:
            epocs = 118287 / batch_size
            boundaries = [epocs * 12, epocs * 19]
        values = [
            learning_rate, learning_rate * 0.5, learning_rate * 0.25
        ]
    elif 'pascalvoc' in data_args.dataset:
        epocs = 19200 / batch_size
        boundaries = [epocs * 40, epocs * 60, epocs * 80, epocs * 100]
        values = [
            learning_rate, learning_rate * 0.5, learning_rate * 0.25,
            learning_rate * 0.1, learning_rate * 0.01
        ]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005), )

    optimizer.minimize(loss)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_gpu, loss_name=loss.name)

    if not args.enable_ce:
        train_reader = paddle.batch(
            reader.train(data_args, train_file_list), batch_size=batch_size)
    else:
        train_reader = paddle.batch(
            reader.train(data_args, train_file_list, False), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, val_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, im_info])

    def save_model(postfix):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print 'save models to %s' % (model_path)
        fluid.io.save_persistables(exe, model_path)
    total_time = 0.0
    for pass_id in range(num_passes):
        epoch_idx = pass_id + 1
        start_time = time.time()
        prev_start_time = start_time
        every_pass_loss = []
        iter = 0
        pass_duration = 0.0
        for batch_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            if len(data) < (devices_num * 2):
                print("There are too few data to train on all devices.")
                continue
            if args.parallel:
                loss_v, = train_exe.run(fetch_list=[loss.name],
                                        feed=feeder.feed(data))
                print(loss_v)
            else:
                loss_v, = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[loss])

            loss_v = np.mean(np.array(loss_v))
            every_pass_loss.append(loss_v)
            if batch_id % 20 == 0:
                print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                    pass_id, batch_id, loss_v, start_time - prev_start_time))

        end_time = time.time()
        if args.enable_ce and pass_id == 1:
            total_time += end_time - start_time
            train_avg_loss = np.mean(every_pass_loss)
            if devices_num == 1:
                print ("kpis    train_cost        %s" % train_avg_loss)
                print ("kpis    train_speed       %s" % (total_time / epoch_idx))
            else:
                print ("kpis    train_cost_card%s   %s" % (devices_num, train_avg_loss))
                print ("kpis    train_speed_card%s  %f" % (devices_num, total_time / epoch_idx))


        if pass_id % 10 == 0 or pass_id == num_passes - 1:
            save_model(str(pass_id))

if __name__ == '__main__':
    print('fasterrcnn')
    args = parser.parse_args()
    print_arguments(args)

    data_dir = args.data_dir
    label_file = 'label_list'
    model_save_dir = args.model_save_dir
    train_file_list = 'trainval.txt'
    val_file_list = 'test.txt'
    if 'coco' in args.dataset:
        data_dir = 'data/coco'
        if '2014' in args.dataset:
            train_file_list = 'annotations/instances_train2014.json'
            val_file_list = 'annotations/instances_val2014.json'
        elif '2017' in args.dataset:
            train_file_list = 'annotations/instances_train2017.json'
            val_file_list = 'annotations/instances_val2017.json'

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        label_file=label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        ap_version = args.ap_version,
        toy=args.is_toy)
    print('train begins...')
    train(
        args,
        train_file_list=train_file_list,
        val_file_list=val_file_list,
        data_args=data_args,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_passes=args.num_passes,
        model_save_dir=model_save_dir,
        pretrained_model=args.pretrained_model)
