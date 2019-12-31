#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from eval_helper import *
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args, check_gpu
import models.model_builder as model_builder
import models.resnet as resnet
from config import cfg
from data_utils2 import DatasetPath
import checkpoint as checkpoint
#from train import *
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def eval():

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
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
            model.build_model(image_shape)
            pred_boxes = model.eval_bbox_out()
    infer_prog = infer_prog.clone(True)
    exe.run(startup_prog)

    # yapf: disable
    def if_exist(var):
        return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
    if cfg.pretrained_model:
        checkpoint.load_params(exe, infer_prog, cfg.pretrained_model)
    # yapf: enable
    test_reader = reader.test(1)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    fetch_list = [pred_boxes]
    res_list = []
    keys = [
        'bbox', 'gt_box', 'gt_class', 'is_crowed', 'im_info', 'im_id',
        'is_difficult'
    ]
    for i, data in enumerate(test_reader()):
        im_info = [data[0][1]]
        result = exe.run(infer_prog,
                         fetch_list=[v.name for v in fetch_list],
                         feed=feeder.feed(data),
                         return_numpy=False)
        pred_boxes_v = result[0]
        nmsed_out = pred_boxes_v
        outs = np.array(nmsed_out)
        res = get_dict(outs, data[0], keys)
        res_list.append(res)
        if i % 50 == 0:
            logger.info('test_iter {}'.format(i))
    icdar_eval(res_list)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    eval()
