import os
import time
import numpy as np
import argparse
import functools
import math

import paddle
import paddle.fluid as fluid
import reader
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('dataset',          str,   'pascalvoc',  "coco2014, coco2017, and pascalvoc.")
add_arg('batch_size',       int,   32,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('data_dir',         str,   '',        "The data root path.")
add_arg('test_list',        str,   '',        "The testing data lists.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('nms_threshold',    float, 0.45,   "NMS threshold.")
add_arg('ap_version',       str,   '11point',   "integral, 11point.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.")  #123.68
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")  #116.78
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")  #103.94
# yapf: enable


def build_program(main_prog, startup_prog, args, data_args):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if 'coco' in data_args.dataset:
        num_classes = 91
    elif 'pascalvoc' in data_args.dataset:
        num_classes = 21

    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=64,
            shapes=[[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1]],
            lod_levels=[0, 1, 1, 1],
            dtypes=["float32", "float32", "int32", "int32"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, gt_box, gt_label, difficult = fluid.layers.read_file(
                py_reader)
            locs, confs, box, box_var = mobile_net(num_classes, image,
                                                   image_shape)
            nmsed_out = fluid.layers.detection_output(
                locs, confs, box, box_var, nms_threshold=args.nms_threshold)
            with fluid.program_guard(main_prog):
                map = fluid.metrics.DetectionMAP(
                    nmsed_out,
                    gt_label,
                    gt_box,
                    difficult,
                    num_classes,
                    overlap_threshold=0.5,
                    evaluate_difficult=False,
                    ap_version=args.ap_version)
    return py_reader, map


def eval(args, data_args, test_list, batch_size, model_dir=None):
    startup_prog = fluid.Program()
    test_prog = fluid.Program()

    test_py_reader, map_eval = build_program(
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args,
        data_args=data_args)
    test_prog = test_prog.clone(for_test=True)
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    def if_exist(var):
        return os.path.exists(os.path.join(model_dir, var.name))

    fluid.io.load_vars(
        exe, model_dir, main_program=test_prog, predicate=if_exist)

    test_reader = reader.test(data_args, test_list, batch_size=batch_size)
    test_py_reader.decorate_paddle_reader(test_reader)

    _, accum_map = map_eval.get_map_var()
    map_eval.reset(exe)
    test_py_reader.start()
    try:
        batch_id = 0
        while True:
            test_map, = exe.run(test_prog, fetch_list=[accum_map])
            if batch_id % 10 == 0:
                print("Batch {0}, map {1}".format(batch_id, test_map))
            batch_id += 1
    except (fluid.core.EOFException, StopIteration):
        test_py_reader.reset()
    print("Test model {0}, map {1}".format(model_dir, test_map))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/pascalvoc'
    test_list = 'test.txt'
    label_file = 'label_list'

    if not os.path.exists(args.model_dir):
        raise ValueError("The model path [%s] does not exist." %
                         (args.model_dir))
    if 'coco' in args.dataset:
        data_dir = 'data/coco'
        if '2014' in args.dataset:
            test_list = 'annotations/instances_val2014.json'
        elif '2017' in args.dataset:
            test_list = 'annotations/instances_val2017.json'

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=args.data_dir if len(args.data_dir) > 0 else data_dir,
        label_file=label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        apply_distort=False,
        apply_expand=False,
        ap_version=args.ap_version)
    eval(
        args,
        data_args=data_args,
        test_list=args.test_list if len(args.test_list) > 0 else test_list,
        batch_size=args.batch_size,
        model_dir=args.model_dir)
