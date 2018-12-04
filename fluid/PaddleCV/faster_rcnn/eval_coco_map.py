#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
from eval_helper import get_nmsed_box
from eval_helper import get_dt_res
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


def eval():
    if '2014' in cfg.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in cfg.dataset:
        test_list = 'annotations/instances_val2017.json'

    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
    class_nums = cfg.class_num
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    total_batch_size = devices_num * cfg.TRAIN.im_per_batch
    cocoGt = COCO(os.path.join(cfg.data_dir, test_list))
    numId_to_catId_map = {i + 1: v for i, v in enumerate(cocoGt.getCatIds())}
    category_ids = cocoGt.getCatIds()
    label_list = {
        item['id']: item['name']
        for item in cocoGt.loadCats(category_ids)
    }
    label_list[0] = ['background']

    model = model_builder.FasterRCNN(
        add_conv_body_func=resnet.add_ResNet50_conv4_body,
        add_roi_box_head_func=resnet.add_ResNet_roi_conv5_head,
        use_pyreader=False,
        is_train=False)
    model.build_model(image_shape)
    rpn_rois, confs, locs = model.eval_bbox_out()
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
    test_reader = reader.test(total_batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    if cfg.MASK_ON:
        fetch_list = [rpn_rois, confs, locs, masks]
    else:
        fetch_list = [rpn_rois, confs, locs]
    for batch_id, batch_data in enumerate(test_reader()):
        start = time.time()
        im_info = []
        for data in batch_data:
            im_info.append(data[1])
        result, = exe.run(fetch_list=[v.name for v in fetch_list],
                          feed=feeder.feed(batch_data),
                          return_numpy=False)
        rpn_rois_v = result[0]
        confs_v = result[1]
        locs_v = result[2]
        if cfg.MASK_ON:
            masks_v = result[3]
        new_lod, nmsed_out = get_nmsed_box(rpn_rois_v, confs_v, locs_v,
                                           class_nums, im_info)

        dts_res += get_dt_res(total_batch_size, new_lod, nmsed_out, batch_data,
                              numId_to_catId_map)
        if cfg.MASK_ON:
            segms_out = segm_results(nmsed_out, masks_v, im_info)
            segms_res = get_segms_res(batch_size, new_lod, segms_out,
                                      batch_data, numId_to_catId_map)
        end = time.time()
        print('batch id: {}, time: {}'.format(batch_id, end - start))
    with open("detection_bbox_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate bbox using coco api")
    cocoDt = cocoGt.loadRes("detection_segms_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if cfg.MASK_ON:
        with open("detection_segms_result.json", 'w') as outfile:
            json.dump(segms_res, outfile)
        print("start evaluate mask using coco api")
        cocoDt = cocoGt.loadRes("detection_segms_result.json")
        cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()
