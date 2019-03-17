# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import box_utils
import reader
from utility import print_arguments, parse_args
from models.yolov3 import YOLOv3
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg


def infer():

    if not os.path.exists('output'):
        os.mkdir('output')

    model = YOLOv3(is_train=False)
    model.build_model()
    outputs = model.get_pred()
    input_size = cfg.input_size
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.weights:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.weights, var.name))
        fluid.io.load_vars(exe, cfg.weights, predicate=if_exist)
    # yapf: enable
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())
    fetch_list = [outputs]
    image_names = []
    if cfg.image_name is not None:
        image_names.append(cfg.image_name)
    else:
        for image_name in os.listdir(cfg.image_path):
            if image_name.split('.')[-1] in ['jpg', 'png']:
                image_names.append(image_name)
    for image_name in image_names:
        infer_reader = reader.infer(input_size, os.path.join(cfg.image_path, image_name))
        label_names, _ = reader.get_label_infos()
        data = next(infer_reader())
        im_shape = data[0][2]
        outputs = exe.run(
            fetch_list=[v.name for v in fetch_list],
            feed=feeder.feed(data),
            return_numpy=False)
        bboxes = np.array(outputs[0])
        if bboxes.shape[1] != 6:
            print("No object found in {}".format(image_name))
            continue
        labels = bboxes[:, 0].astype('int32')
        scores = bboxes[:, 1].astype('float32')
        boxes = bboxes[:, 2:].astype('float32')

        path = os.path.join(cfg.image_path, image_name)
        box_utils.draw_boxes_on_image(path, boxes, scores, labels, label_names, cfg.draw_thresh)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    infer()
