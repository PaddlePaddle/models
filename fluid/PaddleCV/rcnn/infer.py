import os
import time
import numpy as np
from eval_helper import *
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args
import models.model_builder as model_builder
import models.resnet as resnet
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg
from roidbs import DatasetPath


def infer():

    data_path = DatasetPath('val')
    test_list = data_path.get_file_list()

    cocoGt = COCO(test_list)
    num_id_to_cat_id_map = {i + 1: v for i, v in enumerate(cocoGt.getCatIds())}
    category_ids = cocoGt.getCatIds()
    label_list = {
        item['id']: item['name']
        for item in cocoGt.loadCats(category_ids)
    }
    label_list[0] = ['background']
    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
    class_nums = cfg.class_num

    model = model_builder.RCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=False,
        mode='infer')
    model.build_model(image_shape)
    pred_boxes = model.eval_bbox_out()
    if cfg.MASK_ON:
        masks = model.eval_mask_out()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    # yapf: enable
    infer_reader = reader.infer()
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    segms_res = []
    if cfg.MASK_ON:
        fetch_list = [pred_boxes, masks]
    else:
        fetch_list = [pred_boxes]
    data = next(infer_reader())
    im_info = [data[0][1]]
    result = exe.run(fetch_list=[v.name for v in fetch_list],
                     feed=feeder.feed(data),
                     return_numpy=False)
    pred_boxes_v = result[0]
    if cfg.MASK_ON:
        masks_v = result[1]
    new_lod = pred_boxes_v.lod()
    nmsed_out = pred_boxes_v
    path = os.path.join(cfg.image_path, cfg.image_name)
    image = None
    if cfg.MASK_ON:
        segms_out = segm_results(nmsed_out, masks_v, im_info)
        image = draw_mask_on_image(path, segms_out, cfg.draw_threshold)

    draw_bounding_box_on_image(path, nmsed_out, cfg.draw_threshold, label_list,
                               num_id_to_cat_id_map, image)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    infer()
