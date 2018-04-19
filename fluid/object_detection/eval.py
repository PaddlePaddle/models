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
add_arg('dataset',          str,   'coco2014',  "coco2014, coco2017, and pascalvoc.")
add_arg('batch_size',       int,   32,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('data_dir',         str,   '',        "The data root path.")
add_arg('test_list',        str,   '',        "The testing data lists.")
add_arg('model_dir',        str,   '',     "The path to save model.")
add_arg('nms_threshold',    float, 0.5,    "nms threshold")
add_arg('ap_version',       str,   'integral',   "integral, 11points, and cocoMAP")
add_arg('resize_h',         int,   300,    "resize image size")
add_arg('resize_w',         int,   300,    "resize image size")
add_arg('mean_value_B',     float, 127.5, "mean value for B channel which will be subtracted")  #123.68
add_arg('mean_value_G',     float, 127.5, "mean value for G channel which will be subtracted")  #116.78
add_arg('mean_value_R',     float, 127.5, "mean value for R channel which will be subtracted")  #103.94
# yapf: enable


def eval(args, data_args, test_list, batch_size, model_dir=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if 'coco' in data_args.dataset:
        num_classes = 91
    elif 'pascalvoc' in data_args.dataset:
        num_classes = 21

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)
    gt_iscrowd = fluid.layers.data(
        name='gt_iscrowd', shape=[1], dtype='int32', lod_level=1)
    gt_image_info = fluid.layers.data(
        name='gt_image_id', shape=[3], dtype='int32', lod_level=1)

    locs, confs, box, box_var = mobile_net(num_classes, image, image_shape)
    nmsed_out = fluid.layers.detection_output(
        locs, confs, box, box_var, nms_threshold=args.nms_threshold)
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
    if 'cocoMAP' in data_args.ap_version:
        feeder = fluid.DataFeeder(
            place=place, feed_list=[image, gt_box, gt_label, gt_iscrowd, gt_image_info])
    else:
        feeder = fluid.DataFeeder(
            place=place, feed_list=[image, gt_box, gt_label, difficult])

    def test():
        if 'cocoMAP' in data_args.ap_version:
            dts_res = []
            import json

            for batch_id, data in enumerate(test_reader()):
                nmsed_out_v = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[nmsed_out],
                                        return_numpy=False)
                if batch_id % 20 == 0:
                    print("Batch {0}".format(batch_id))

                lod = nmsed_out_v[0].lod()[0]
                nmsed_out_v = np.array(nmsed_out_v[0])
                real_batch_size = min(batch_size, len(data))
                assert (len(lod) == real_batch_size + 1), \
                "Error Lod Tensor offset dimension. Lod({}) vs. batch_size({})".format(len(lod), batch_size)
                k = 0
                for i in range(real_batch_size):
                    dt_num_this_img = lod[i + 1] - lod[i]
                    image_id = int(data[i][4][0])
                    image_width = int(data[i][4][1])
                    image_height = int(data[i][4][2])
                    for j in range(dt_num_this_img):
                        dt = nmsed_out_v[k]
                        k = k + 1
                        category_id, score, xmin, ymin, xmax, ymax = dt.tolist()
                        xmin = max(min(xmin, 1.0), 0.0) * image_width
                        ymin = max(min(ymin, 1.0), 0.0) * image_height
                        xmax = max(min(xmax, 1.0), 0.0) * image_width
                        ymax = max(min(ymax, 1.0), 0.0) * image_height
                        w = xmax - xmin
                        h = ymax - ymin
                        bbox = [xmin, ymin, w, h]
                        dt_res = {
                            'image_id' : image_id,
                            'category_id' : category_id,
                            'bbox' : bbox,
                            'score' : score
                        }
                        dts_res.append(dt_res)
            
            with open("detection_result.json", 'w') as outfile:
                json.dump(dts_res, outfile)
            print("start evaluate using coco api")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            cocoGt=COCO(os.path.join(args.data_dir,args.val_file_list))
            cocoDt=cocoGt.loadRes("detection_result.json")
            cocoEval = COCOeval(cocoGt,cocoDt,"bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

        else:
            _, accum_map = map_eval.get_map_var()
            map_eval.reset(exe)
            for batch_id, data in enumerate(test_reader()):
                test_map = exe.run(test_program,
                                   feed=feeder.feed(data),
                                   fetch_list=[accum_map])
                if batch_id % 20 == 0:
                    print("Batch {0}, map {1}".format(idx, test_map[0]))
            print("Test model {0}, map {1}".format(model_dir, test_map[0]))
    test()

if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/pascalvoc'
    test_list = 'test.txt'
    label_file = 'label_list'
    if 'coco' in args.dataset:
        data_dir = './data/coco'
        if '2014' in args.dataset:
            test_list = 'annotations/instances_minival2014.json'
        elif '2017' in args.dataset:
            test_list = 'annotations/instances_val2017.json'

    data_args = reader.Settings(
        dataset=args.dataset,
        ap_version = args.ap_version,
        toy=0,
        data_dir=data_dir,
        label_file=label_file,
        apply_distort=False,
        apply_expand=False,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R])
    eval(
        args,
        test_list=args.test_list,
        data_args=data_args,
        batch_size=args.batch_size,
        model_dir=args.model_dir)