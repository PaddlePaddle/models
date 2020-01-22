#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
import json
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import reader
from models.yolov3 import YOLOv3
from utility import print_arguments, parse_args, check_gpu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg


def eval():
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)

    if '2014' in cfg.dataset:
        test_list = 'annotations/instances_val2014.json'
    elif '2017' in cfg.dataset:
        test_list = 'annotations/instances_val2017.json'

    if cfg.debug:
        if not os.path.exists('output'):
            os.mkdir('output')

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = YOLOv3(3,is_train=False)
        # yapf: disable
        if cfg.weights:
            restore, _ = fluid.load_dygraph(cfg.weights)
            model.set_dict(restore)
            model.eval()

        input_size = cfg.input_size
        # batch_size for test must be 1
        test_reader = reader.test(input_size, 1)
        label_names, label_ids = reader.get_label_infos()
        if cfg.debug:
            print("Load in labels {} with ids {}".format(label_names, label_ids))

        def get_pred_result(boxes, scores, labels, im_id):
            result = []
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                bbox = [x1, y1, w, h]

                res = {
                    'image_id': int(im_id),
                    'category_id': label_ids[int(label)],
                    'bbox': list(map(float, bbox)),
                    'score': float(score)
                }
                result.append(res)
            return result

        dts_res = []
        total_time = 0
        for iter_id, data in enumerate(test_reader()):
            start_time = time.time()

            img_data = np.array([x[0] for x in data]).astype('float32')
            img = to_variable(img_data)

            im_id_data = np.array([x[1] for x in data]).astype('int32')
            im_id = to_variable(im_id_data)

            im_shape_data = np.array([x[2] for x in data]).astype('int32')
            im_shape = to_variable(im_shape_data)

            batch_outputs = model(img, None, None, None, im_id, im_shape)
            nmsed_boxes = batch_outputs.numpy()
            if nmsed_boxes.shape[1] != 6:
                continue

            im_id = data[0][1]
            nmsed_box=nmsed_boxes
            labels = nmsed_box[:, 0]
            scores = nmsed_box[:, 1]
            boxes = nmsed_box[:, 2:6]
            dts_res += get_pred_result(boxes, scores, labels, im_id)

            end_time = time.time()
            print("batch id: {}, time: {}".format(iter_id, end_time - start_time))
            total_time += end_time - start_time

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

    print("Time per batch: {}".format(total_time / iter_id))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()

