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


def eval(cfg):

    if '2014' in cfg.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in cfg.dataset:
        test_list = 'annotations/instances_val2017.json'

    image_shape = [3, cfg.max_size, cfg.max_size]
    class_nums = cfg.class_num
    batch_size = cfg.batch_size
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
        is_train=False)
    model.build_model(image_shape)
    rpn_rois, confs, locs = model.eval_out()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    # yapf: enable
    test_reader = reader.test(cfg, batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    fetch_list = [rpn_rois, confs, locs]
    for batch_id, batch_data in enumerate(test_reader()):
        start = time.time()
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

        dts_res += get_dt_res(batch_size, new_lod, nmsed_out, batch_data)
        end = time.time()
        print('batch id: {}, time: {}'.format(batch_id, end - start))
    with open("detection_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate using coco api")
    cocoDt = cocoGt.loadRes("detection_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)

    data_args = reader.Settings(args)
    eval(data_args)
