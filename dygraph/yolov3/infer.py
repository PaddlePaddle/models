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
from utility import print_arguments, parse_args, check_gpu
from models.yolov3 import YOLOv3
from paddle.fluid.dygraph.base import to_variable
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg


def infer():

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)

    if not os.path.exists('output'):
        os.mkdir('output')
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):

        model = YOLOv3(3, is_train=False)
        input_size = cfg.input_size
        # yapf: disable
        if cfg.weights:
            restore, _ = fluid.load_dygraph(cfg.weights)
            model.set_dict(restore)
        # yapf: enable

        # you can save inference model by following code
        # fluid.io.save_inference_model("./output/yolov3",
        #                               feeded_var_names=['image', 'im_shape'],
        #                               target_vars=outputs,
        #                               executor=exe)

        image_names = []
        if cfg.image_name is not None:
            image_names.append(cfg.image_name)
        else:
            for image_name in os.listdir(cfg.image_path):
                if image_name.split('.')[-1] in ['jpg', 'png']:
                    image_names.append(image_name)
        for image_name in image_names:
            infer_reader = reader.infer(input_size,
                                        os.path.join(cfg.image_path, image_name))
            label_names, _ = reader.get_label_infos()
            data = next(infer_reader())

            img_data = np.array([x[0] for x in data]).astype('float32')
            img = to_variable(img_data)

            im_shape_data = np.array([x[2] for x in data]).astype('int32')
            im_shape = to_variable(im_shape_data)

            outputs = model(img, None, None, None, None, im_shape)

            bboxes = outputs.numpy()
            if bboxes.shape[1] != 6:
                print("No object found in {}".format(image_name))
                continue
            labels = bboxes[:, 0].astype('int32')
            scores = bboxes[:, 1].astype('float32')
            boxes = bboxes[:, 2:].astype('float32')

            path = os.path.join(cfg.image_path, image_name)
            box_utils.draw_boxes_on_image(path, boxes, scores, labels, label_names,
                                          cfg.draw_thresh)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    infer()
