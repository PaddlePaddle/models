import os
import time
import numpy as np
import argparse
import functools
import shutil
import load_var
import paddle
import paddle.fluid as fluid
import reader
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
from fasterrcnn_model import FasterRcnn, RPNloss
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.01,     "Learning rate.")
add_arg('batch_size',       int,   1,          "Minibatch size.")
add_arg('parallel',         bool,  False,       "Minibatch size.")
add_arg('num_passes',       int,   1,        "Epoch number.")
add_arg('use_gpu',          bool,  False,      "Whether use GPU.")
add_arg('dataset',          str,   'pascalvoc', "coco2014, coco2017, and pascalvoc.")
add_arg('model_save_dir',   str,   'model',     "The path to save model.")
add_arg('pretrained_model', str,   'imagenet_resnet50', "The init model path.")
add_arg('anchor_sizes',      int,   [32,64,128,256,512],  "The size of anchors.")
add_arg('aspect_ratios',    float,   [0.5,1.0,2.0],    "The ratio of anchors.")
add_arg('resize_h',         int,   800,    "The resized image height.")
add_arg('resize_w',         int,   1333,    "The resized image height.")
add_arg('data_dir',         str,   'data/pascalvoc', "data directory")
#yapf: enable

def exponential_with_warmup_decay(boundaries, values, warmup_iter, warmup_factor):
    global_step = lr_scheduler._decay_step_counter()
    decayed_lr = lr_scheduler.piecewise_decay(boundaries, values)
    with control_flow.Switch() as switch:
        with switch.case(global_step < warmup_iter):
            alpha = global_step / warmup_iter
            factor = warmup_factor * (1 - alpha) + alpha
            decayed_lr = decayed_lr * factor
    return decayed_lr

def train(args,
          train_file_list,
          val_file_list,
          data_args,
          learning_rate,
          batch_size,
          num_passes,
          model_save_dir,
          pretrained_model=None):

    image_shape = [3, data_args.resize_h, data_args.resize_w]
    class_nums = 81

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1],  dtype='int32', lod_level=1)
    is_crowd = fluid.layers.data(
        name='is_crowd', shape = [1], dtype='int32', lod_level=1)
    im_info = fluid.layers.data(
        name='im_info', shape=[3], dtype='float32')

    rpn_cls_score, rpn_bbox_pred, anchor, var, cls_score, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois, labels_int32 = FasterRcnn(
            input=image,
            depth=50,
            anchor_sizes=args.anchor_sizes,
            variance=[10.,10.,5.,5.],
            aspect_ratios=args.aspect_ratios,
            gt_box=gt_box,
            is_crowd=is_crowd,
            gt_label=gt_label,
            im_info=im_info,
            class_nums=class_nums,
            )
    cls_loss, reg_loss = RPNloss(rpn_cls_score, rpn_bbox_pred, anchor, var, gt_box, is_crowd, im_info)
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
    cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
    loss_cls = fluid.layers.cross_entropy(cls_prob, labels_int64)
    loss_cls = fluid.layers.reduce_mean(loss_cls)
    loss_bbox = fluid.layers.smooth_l1(x=bbox_pred,
                                        y=bbox_targets,
                                        inside_weight=bbox_inside_weights,
                                        outside_weight=bbox_outside_weights,
                                        sigma=1.0)
    loss_bbox = fluid.layers.reduce_mean(loss_bbox)

    loss_cls.persistable=True
    loss_bbox.persistable=True
    detection_loss = loss_cls + loss_bbox
    detection_loss.persistable=True

    loss = rpn_loss + detection_loss
    loss.persistable=True

    boundaries = [120000,160000]
    values = [learning_rate,learning_rate*0.1,learning_rate*0.01]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.0001),
        momentum=0.9)
    optimizer.minimize(loss)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    if pretrained_model:
        def if_exist(var):
            #if (os.path.exists(os.path.join(pretrained_model, var.name))):
            #	print(var.name)
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    if args.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_gpu, loss_name=loss.name)

    train_reader = paddle.batch(
        reader.train(data_args, train_file_list,False), batch_size=batch_size)
    test_reader = paddle.batch(
        reader.test(data_args, val_file_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, is_crowd, im_info])
    def save_model(postfix):
        model_path = os.path.join(model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        print 'save models to %s' % (model_path)
        fluid.io.save_persistables(exe, model_path)
    total_time = 0.0
    for pass_id in range(num_passes):
        start_time = time.time()
        prev_start_time = start_time
        every_pass_loss = []
        iter = 0
        pass_duration = 0.0
        for batch_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            if args.parallel:
                loss_v, = train_exe.run(fetch_list=[loss.name],
                                        feed=feeder.feed(data))
                print(loss_v)
            else:
                loss_v, = exe.run(fluid.default_main_program(),
                                                feed=feeder.feed(data),
                                                fetch_list=[loss])
                print(loss_v)

            loss_v = np.mean(np.array(loss_v))
            every_pass_loss.append(loss_v)
            if batch_id % 20 == 0:
                print("Pass {0}, batch {1}, loss {2}, time {3}".format(
                    pass_id, batch_id, loss_v, start_time - prev_start_time))
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
    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=data_dir,
        label_file=label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w)
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
