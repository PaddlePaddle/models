import os
import time
import numpy as np
import argparse
import functools
from eval_helper import get_nmsed_box
from eval_helper import get_dt_res
from eval_helper import draw_bounding_box_on_image
import paddle
import paddle.fluid as fluid
import reader
from utility import add_arguments, print_arguments
# A special mAP metric for COCO dataset, which averages AP in different IoUs.
# To use this eval_cocoMAP.py, [cocoapi](https://github.com/cocodataset/cocoapi) is needed.
import models.model_builder as model_builder
import models.resnet as resnet
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
# ENV
add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('dataset',          str,   'coco2017',  "coco2014, coco2017.")
add_arg('data_dir',         str,   'data/COCO17',        "The data root path.")
add_arg('class_num',        int,   81,          "Class number.")
# RPN
add_arg('anchor_sizes',     int,    [32,64,128,256,512],  "The size of anchors.")
add_arg('aspect_ratios',    float,  [0.5,1.0,2.0],    "The ratio of anchors.")
add_arg('variance',         float,  [1.,1.,1.,1.],    "The variance of anchors.")
# FAST RCNN
# EVAL
add_arg('batch_size',       int,   1,        "Minibatch size.")
add_arg('max_size',         int,   1333,    "The resized image height.")
add_arg('scales', int,  [800],    "The resized image height.")
add_arg('mean_value',     float,   [102.9801, 115.9465, 122.7717], "pixel mean")
add_arg('nms_threshold',    float, 0.5,    "NMS threshold.")
add_arg('score_threshold',    float, 0.05,    "score threshold for NMS.")
# SINGLE EVAL AND DRAW
add_arg('one_eval',     bool,    False,    "Whether evaluate  one image ")
add_arg('draw_threshold',  float, 0.8,    "Confidence threshold to draw bbox.")
add_arg('image_path',       str,   'data/COCO17/val2017',  "The image path used to inference and visualize.")
add_arg('image_name',        str,    '',       "The single image used to inference and visualize.")
# yapf: enable


def eval(cfg):

    if '2014' in cfg.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in cfg.dataset:
        test_list = 'annotations/instances_val2017.json'

    image_shape = [3, cfg.max_size, cfg.max_size]
    class_nums = cfg.class_num
    batch_size = cfg.batch_size
    if cfg.one_eval:
        assert batch_size == 1
    cocoGt = COCO(os.path.join(cfg.data_dir, test_list))
    numId_to_catId_map = {i + 1: v for i, v in enumerate(cocoGt.getCatIds())}
    category_ids = cocoGt.getCatIds()
    label_list = {
        item['id']: item['name']
        for item in cocoGt.loadCats(category_ids)
    }
    label_list[0] = ['background']

    model = model_builder.FasterRCNN(
        cfg=cfg,
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=False,
        is_train=False,
        use_random=False)
    model.build_model(image_shape)
    rpn_rois, confs, locs = model.eval_out()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.model_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.model_dir, var.name))
        fluid.io.load_vars(exe, cfg.model_dir, predicate=if_exist)
    # yapf: enable
    test_reader = reader.test(cfg, batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    fetch_list = [rpn_rois, confs, locs]
    for batch_id, batch_data in enumerate(test_reader()):
        start = time.time()
        #image, im_info, im_id = batch_data[0]
        im_info = []
        for data in batch_data:
            im_info.append(data[1])
        rpn_rois_v, confs_v, locs_v = exe.run(
            fetch_list=[v.name for v in fetch_list],
            feed=feeder.feed(batch_data),
            return_numpy=False)
        new_lod, nmsed_out = get_nmsed_box(cfg, rpn_rois_v, confs_v, locs_v,
                                           class_nums, im_info,
                                           numId_to_catId_map)
        for data in batch_data:
            if str(data[-1]) in cfg.image_name:
                path = os.path.join(cfg.image_path, cfg.image_name)
                draw_bounding_box_on_image(path, nmsed_out, cfg.draw_threshold,
                                           label_list)

        dts_res += get_dt_res(batch_size, new_lod, nmsed_out, batch_data)
        end = time.time()
        print('batch id: {}, time: {}'.format(batch_id, end - start))
        if cfg.one_eval:
            print('evaluate one image: {}'.format(cfg.image_name))
            break
    with open("detection_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate using coco api")
    cocoDt = cocoGt.loadRes("detection_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    if cfg.one_eval:
        cocoEval.params.imgIds = batch_data[0][-1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_args = reader.Settings(args)
    eval(data_args)
