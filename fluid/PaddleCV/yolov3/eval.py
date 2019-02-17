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
import paddle
import paddle.fluid as fluid
import box_utils
import reader
import models
from utility import print_arguments, parse_args
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config.config import cfg


def eval():
    if '2014' in cfg.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in cfg.dataset:
        test_list = 'annotations/instances_val2017.json'

    if cfg.debug:
        if not os.path.exists('output'):
            os.mkdir('output')

    model = models.YOLOv3(cfg.model_cfg_path, is_train=False)
    model.build_model()
    outputs = model.get_pred()
    hyperparams = model.get_hyperparams()
    yolo_anchors = model.get_yolo_anchors()
    yolo_classes = model.get_yolo_classes()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    # yapf: enable
    input_size = model.get_input_size()
    test_reader = reader.test(input_size, 1)
    label_names, label_ids = reader.get_label_infos()
    if cfg.debug:
        print("Load in labels {} with ids {}".format(label_names, label_ids))
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    def get_pred_result(boxes, scores, labels, im_id):
        result = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            bbox = [x1, y1, w, h]
            
            res = {
                    'image_id': im_id,
                    'category_id': label_ids[int(label)],
                    'bbox': bbox,
                    'score': score
            }
            result.append(res)
        return result

    dts_res = []
    fetch_list = outputs
    total_time = 0
    for batch_id, batch_data in enumerate(test_reader()):
        start_time = time.time()
        batch_outputs = exe.run(
            fetch_list=[v.name for v in fetch_list],
            feed=feeder.feed(batch_data),
            return_numpy=False)
        for data, outputs in zip(batch_data, batch_outputs):
            im_id = data[1]
            im_shape = data[2]
            pred_boxes, pred_scores, pred_labels = box_utils.get_all_yolo_pred(
                    batch_outputs, yolo_anchors, yolo_classes, (input_size, input_size))
            boxes, scores, labels = box_utils.calc_nms_box_new(pred_boxes, pred_scores, pred_labels,
                                                    cfg.valid_thresh, cfg.nms_thresh)
            boxes = box_utils.rescale_box_in_input_image(boxes, im_shape, input_size)
            dts_res += get_pred_result(boxes, scores, labels, im_id)
            end_time = time.time()
            print("batch id: {}, time: {}".format(batch_id, end_time - start_time))
            total_time += (end_time - start_time)

            if cfg.debug:
                if '2014' in cfg.dataset:
                    img_name = "COCO_val2014_{:012d}.jpg".format(im_id)
                    box_utils.draw_boxes_on_image(os.path.join("./dataset/coco/val2014", img_name), boxes, scores, labels, label_names)
                if '2017' in cfg.dataset:
                    img_name = "{:012d}.jpg".format(im_id)
                    box_utils.draw_boxes_on_image(os.path.join("./dataset/coco/val2017", img_name), boxes, scores, labels, label_names)

    with open("yolov3_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate detection result with coco api")
    coco = COCO(os.path.join(cfg.data_dir, test_list))
    cocoDt = coco.loadRes("yolov3_result.json")
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("evaluate done.")

    print("Time per batch: {}".format(total_time / batch_id))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()
