# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import io
import json
import sys
import re
import numpy as np
import time
import shutil
import collections
import six
import pickle
import paddle.fluid as fluid
import reader
from models.dyg.model_builder import RCNN
from config import cfg
from utility import parse_args, print_arguments, SmoothedValue, TrainingStats, now_time, check_gpu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from data_utils import DatasetPath
from eval_helper import (get_dt_res, segm_results, get_segms_res)


def eval():
    devices_num = 1
    total_batch_size = 1  #devices_num * cfg.TRAIN.im_per_batch

    data_path = DatasetPath('val')
    test_list = data_path.get_file_list()
    cocoGt = COCO(test_list)
    num_id_to_cat_id_map = {i + 1: v for i, v in enumerate(cocoGt.getCatIds())}

    use_random = True
    if cfg.enable_ce:
        use_random = False

    if cfg.parallel:
        strategy = fluid.dygraph.parallel.prepare_context()
        print("Execute Parallel Mode!!!")

    # Model
    model = RCNN("faster_rcnn", cfg=cfg, mode='eval', use_random=use_random)

    if cfg.parallel:
        model = fluid.dygraph.parallel.DataParallel(model, strategy)

    if False:  #cfg.pretrained_model:
        model_state = model.state_dict()
        ckpt_file = open(cfg.pretrained_model, 'r')
        w_dict = pickle.load(ckpt_file)
        for k, v in w_dict.items():
            for wk in model_state.keys():
                res = re.search(k, wk)
                if res is not None:
                    print("load: ", k, v.shape, np.mean(np.abs(v)), " --> ", wk,
                          model_state[wk].shape)
                    model_state[wk] = v
                    break
        model.set_dict(model_state)
    elif cfg.resume_model:
        para_state_dict, opti_state_dict = fluid.load_dygraph("model_final")
        #print(para_state_dict.keys())
        #ckpt_file = open("dyg_mask_rcnn.pkl", "w")
        new_dict = {}
        for k, v in para_state_dict.items():
            if "conv2d" in k:
                new_k = k.split('.')[1]
            elif 'linear' in k:
                new_k = k.split('.')[1]
            elif 'conv2dtranspose' in k:
                new_k = k.split('.')[1]
            else:
                new_k = k
            print("save weight from %s to %s" % (k, new_k))
            new_dict[new_k] = v.numpy()
        #print(new_dict.keys())
        #pickle.dump(new_dict, ckpt_file)
        np.savez("dyg_mask_rcnn.npz", **new_dict)
        model.set_dict(para_state_dict)

    test_reader = reader.test(batch_size=total_batch_size)
    if cfg.parallel:
        train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)

    eval_start = time.time()
    dts_res = []
    segms_res = []
    for iter_id, data in enumerate(test_reader()):
        start = time.time()

        image_data = np.array([x[0] for x in data]).astype('float32')
        image_info_data = np.array([x[1] for x in data]).astype('float32')
        image_id_data = np.array([x[2] for x in data]).astype('int32')

        if cfg.enable_ce:
            print("image_data: ", np.abs(image_data).mean(), image_data.shape)
            print("im_info_dta: ", np.abs(image_info_data).mean(),
                  image_info_data.shape, image_info_data)
            print("img_id: ", image_id_data, image_id_data.shape)

        # forward
        outputs = model(image_data, image_info_data, image_id_data)

        pred_boxes_v = outputs[1].numpy()
        if cfg.MASK_ON:
            masks_v = outputs[2].numpy()

        new_lod = list(outputs[0].numpy())
        #new_lod = [[0, pred_boxes_v.shape[0]]] #pred_boxes_v.lod()
        nmsed_out = pred_boxes_v

        dts_res += get_dt_res(total_batch_size, new_lod, nmsed_out, data,
                              num_id_to_cat_id_map)

        if cfg.MASK_ON and np.array(masks_v).shape != (1, 1):
            segms_out = segm_results(nmsed_out, masks_v, image_info_data)
            segms_res += get_segms_res(total_batch_size, new_lod, segms_out,
                                       data, num_id_to_cat_id_map)

        end = time.time()
        print('batch id: {}, time: {}'.format(iter_id, end - start))
    eval_end = time.time()
    total_time = eval_end - eval_start
    print('average time of eval is: {}'.format(total_time / (iter_id + 1)))
    assert len(dts_res) > 0, "The number of valid bbox detected is zero.\n \
        Please use reasonable model and check input data."

    if cfg.MASK_ON:
        assert len(
            segms_res) > 0, "The number of valid mask detected is zero.\n \
            Please use reasonable model and check input data."

    with io.open("detection_bbox_result.json", 'w') as outfile:
        encode_func = unicode if six.PY2 else str
        outfile.write(encode_func(json.dumps(dts_res)))
    print("start evaluate bbox using coco api")
    cocoDt = cocoGt.loadRes("detection_bbox_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if cfg.MASK_ON:
        with io.open("detection_segms_result.json", 'w') as outfile:
            encode_func = unicode if six.PY2 else str
            outfile.write(encode_func(json.dumps(segms_res)))
        print("start evaluate mask using coco api")
        cocoDt = cocoGt.loadRes("detection_segms_result.json")
        cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if cfg.parallel else fluid.CUDAPlace(0) \
        if cfg.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        eval()
