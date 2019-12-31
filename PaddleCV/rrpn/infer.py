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
from eval_helper import clip_box


def infer():

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
    class_nums = cfg.class_num
    model = model_builder.RRPN(
        add_conv_body_func=resnet.ResNet(),
        add_roi_box_head_func=resnet.ResNetC5(),
        use_pyreader=False,
        mode='infer')

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
    infer_reader = reader.infer(cfg.image_path)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    fetch_list = [pred_boxes]
    imgs = os.listdir(cfg.image_path)
    imgs.sort()

    for i, data in enumerate(infer_reader()):
        result = exe.run(infer_prog,
                         fetch_list=[v.name for v in fetch_list],
                         feed=feeder.feed(data),
                         return_numpy=False)
        nmsed_out = result[0]
        im_info = data[0][1]
        im_scale = im_info[2]
        outs = np.array(nmsed_out)
        draw_bounding_box_on_image(cfg.image_path, imgs[i], outs, im_scale,
                                   cfg.draw_threshold)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    infer()
