#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import time
import numpy as np
import pickle
import paddle
import paddle.fluid as fluid
import reader
import models.model_builder as model_builder
import models.resnet as resnet
import checkpoint as checkpoint
from config import cfg
from utility import print_arguments, parse_args, check_gpu
from data_utils import DatasetPath
from eval_helper import *
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def eval():

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    class_nums = cfg.class_num
    model = model_builder.RRPN(
        add_conv_body_func=resnet.ResNet(),
        add_roi_box_head_func=resnet.ResNetC5(),
        use_pyreader=False,
        mode='val')

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            model.build_model()
            pred_boxes = model.eval_bbox_out()
    infer_prog = infer_prog.clone(True)
    exe.run(startup_prog)
    fluid.load(infer_prog, cfg.pretrained_model, exe)
    test_reader = reader.test(1)
    data_loader = model.data_loader
    data_loader.set_sample_list_generator(test_reader, places=place)

    fetch_list = [pred_boxes]
    res_list = []
    keys = [
        'bbox', 'gt_box', 'gt_class', 'is_crowed', 'im_info', 'im_id',
        'is_difficult'
    ]
    for i, data in enumerate(data_loader()):
        result = exe.run(infer_prog,
                         fetch_list=[v.name for v in fetch_list],
                         feed=data,
                         return_numpy=False)
        pred_boxes_v = result[0]
        nmsed_out = pred_boxes_v
        outs = np.array(nmsed_out)
        res = get_key_dict(outs, data[0], keys)
        res_list.append(res)
        if i % 50 == 0:
            logger.info('test_iter {}'.format(i))
    icdar_eval(res_list)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    eval()
