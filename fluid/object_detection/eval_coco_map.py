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

# A special mAP metric for COCO dataset, which averages AP in different IoUs.
# To use this eval_cocoMAP.py, [cocoapi](https://github.com/cocodataset/cocoapi) is needed.
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('dataset',          str,   'coco2014',  "coco2014, coco2017.")
add_arg('batch_size',       int,   32,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('data_dir',         str,   '',        "The data root path.")
add_arg('test_list',        str,   '',        "The testing data lists.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('nms_threshold',    float, 0.5,    "NMS threshold.")
add_arg('ap_version',       str,   'cocoMAP',   "cocoMAP.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.")  #123.68
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")  #116.78
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")  #103.94
# yapf: enable


def eval(args, data_args, test_list, batch_size, model_dir=None):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    num_classes = 91

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    gt_iscrowd = fluid.layers.data(
        name='gt_iscrowd', shape=[1], dtype='int32', lod_level=1)
    gt_image_info = fluid.layers.data(
        name='gt_image_id', shape=[3], dtype='int32', lod_level=1)

    locs, confs, box, box_var = mobile_net(num_classes, image, image_shape)
    nmsed_out = fluid.layers.detection_output(
        locs, confs, box, box_var, nms_threshold=args.nms_threshold)
    loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box, box_var)
    loss = fluid.layers.reduce_sum(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if model_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(model_dir, var.name))
        fluid.io.load_vars(exe, model_dir, predicate=if_exist)
    # yapf: enable
    test_reader = paddle.batch(
        reader.test(data_args, test_list), batch_size=batch_size)
    feeder = fluid.DataFeeder(
        place=place,
        feed_list=[image, gt_box, gt_label, gt_iscrowd, gt_image_info])

    def get_dt_res(nmsed_out_v, data):
        dts_res = []
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
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': bbox,
                    'score': score
                }
                dts_res.append(dt_res)
        return dts_res

    def test():
        dts_res = []

        for batch_id, data in enumerate(test_reader()):
            nmsed_out_v = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[nmsed_out],
                                  return_numpy=False)
            if batch_id % 20 == 0:
                print("Batch {0}".format(batch_id))
            dts_res += get_dt_res(nmsed_out_v, data)

        with open("detection_result.json", 'w') as outfile:
            json.dump(dts_res, outfile)
        print("start evaluate using coco api")
        cocoGt = COCO(os.path.join(data_args.data_dir, test_list))
        cocoDt = cocoGt.loadRes("detection_result.json")
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    test()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = './data/coco'
    if '2014' in args.dataset:
        test_list = 'annotations/instances_minival2014.json'
    elif '2017' in args.dataset:
        test_list = 'annotations/instances_val2017.json'

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=args.data_dir if len(args.data_dir) > 0 else data_dir,
        label_file='',
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        apply_distort=False,
        apply_expand=False,
        ap_version=args.ap_version,
        toy=0)
    eval(
        args,
        data_args=data_args,
        test_list=args.test_list if len(args.test_list) > 0 else test_list,
        batch_size=args.batch_size,
        model_dir=args.model_dir)
