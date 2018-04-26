import os
import time
import numpy as np
import argparse
import functools

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('dataset',          str, 'pascalvoc', "coco or pascalvoc.")
add_arg('batch_size',       int,   32,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,      "Whether to use GPU or not.")
add_arg('data_dir',         str,   '',        "The data root path.")
add_arg('test_list',        str,   '',        "The testing data lists.")
add_arg('label_file',       str,   '',        "The label file, which save the real name and is only used for Pascal VOC.")
add_arg('model_dir',        str,   '',        "The model path.")
add_arg('ap_version',       str,  '11point',  "11point or integral")
add_arg('resize_h',         int,  300,         "The resized image height.")
add_arg('resize_w',         int,  300,         "The resized image width.")
add_arg('mean_value_B',     float, 127.5,      "mean value for B channel which will be subtracted")  #123.68
add_arg('mean_value_G',     float, 127.5,      "mean value for G channel which will be subtracted")  #116.78
add_arg('mean_value_R',     float, 127.5,      "mean value for R channel which will be subtracted")  #103.94
# yapf: enable


def eval(args, data_args, test_list, batch_size, model_dir=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if data_args.dataset == 'coco':
        num_classes = 81
    elif data_args.dataset == 'pascalvoc':
        num_classes = 21

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)

    locs, confs, box, box_var = mobile_net(num_classes, image, image_shape)
    nmsed_out = fluid.layers.detection_output(
        locs, confs, box, box_var, nms_threshold=0.45)
    loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box, box_var)
    loss = fluid.layers.reduce_sum(loss)

    test_program = fluid.default_main_program().clone(for_test=True)
    with fluid.program_guard(test_program):
        map_eval = fluid.evaluator.DetectionMAP(
            nmsed_out,
            gt_label,
            gt_box,
            difficult,
            num_classes,
            overlap_threshold=0.5,
            evaluate_difficult=False,
            ap_version=args.ap_version)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    if model_dir:

        def if_exist(var):
            return os.path.exists(os.path.join(model_dir, var.name))

        fluid.io.load_vars(exe, model_dir, predicate=if_exist)

    test_reader = paddle.batch(
        reader.test(data_args, test_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, difficult])

    _, accum_map = map_eval.get_map_var()
    map_eval.reset(exe)
    for idx, data in enumerate(test_reader()):
        test_map = exe.run(test_program,
                           feed=feeder.feed(data),
                           fetch_list=[accum_map])
        if idx % 50 == 0:
            print("Batch {0}, map {1}".format(idx, test_map[0]))
    print("Test model {0}, map {1}".format(model_dir, test_map[0]))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=args.data_dir,
        label_file=args.label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R])
    eval(
        args,
        test_list=args.test_list,
        data_args=data_args,
        batch_size=args.batch_size,
        model_dir=args.model_dir)
