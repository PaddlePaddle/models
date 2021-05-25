# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import cv2
import numpy as np
import paddle

from smoke.cvlibs import  Config
from smoke.utils import logger, load_pretrained_model
from smoke.utils.vis_utils import get_img, get_ratio, encode_box3d, draw_box_3d

def parse_args():
    parser = argparse.ArgumentParser(description='Model test')

    # params of evaluate
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", required=True, type=str)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for evaluation',
        type=str,
        required=True)
    parser.add_argument(
        '--input_path',
        dest='input_path',
        help='The image path',
        type=str,
        required=True)
    parser.add_argument(
        '--output_path',
        dest='output_path',
        help='The result path of image',
        type=str,
        required=True)
   

    return parser.parse_args()


def main(args):
    
    paddle.set_device("gpu")

    cfg = Config(args.cfg)
   
    model = cfg.model
    model.eval()
    if args.model_path:
        load_pretrained_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')
    K = np.array([[[2055.56, 0, 939.658], [0, 2055.56, 641.072], [0, 0, 1]]], np.float32)
    K_inverse = np.linalg.inv(K)
    K_inverse = paddle.to_tensor(K_inverse)

    img, ori_img_size, output_size = get_img(args.input_path)

    ratio = get_ratio(ori_img_size, output_size)
    ratio = paddle.to_tensor(ratio)
    cam_info = [K_inverse, ratio]
    total_pred = model(img, cam_info)

    keep_idx = paddle.nonzero(total_pred[:, -1] > 0.25)
    total_pred = paddle.gather(total_pred, keep_idx)

    if total_pred.shape[0] > 0:
        pred_dimensions = total_pred[:, 6:9]
        pred_dimensions = pred_dimensions.roll(shifts=1, axis=1)
        pred_rotys = total_pred[:, 12]
        pred_locations = total_pred[:, 9:12]
        bbox_3d = encode_box3d(pred_rotys, pred_dimensions, pred_locations, paddle.to_tensor(K), (1280, 1920))
    else:
        bbox_3d = total_pred
    
    img_draw = cv2.imread(args.input_path)
    for idx in range(bbox_3d.shape[0]):
        bbox = bbox_3d[idx]
        bbox = bbox.transpose([1,0]).numpy()
        img_draw = draw_box_3d(img_draw, bbox)
    
    cv2.imwrite(args.output_path, img_draw)


if __name__ == '__main__':
    args = parse_args()
    main(args)
