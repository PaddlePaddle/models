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

import numpy as np
import argparse

import cv2
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from smoke.utils.vis_utils import encode_box3d, draw_box_3d

def get_ratio(ori_img_size, output_size, down_ratio=(4, 4)):
    return np.array([[down_ratio[1] * ori_img_size[1] / output_size[1], 
                     down_ratio[0] * ori_img_size[0] / output_size[0]]], np.float32)

def get_img(img_path):
    img = cv2.imread(img_path)
    ori_img_size = img.shape
    img = cv2.resize(img, (960, 640))
    output_size = img.shape
    img = img/255.0
    img = np.subtract(img, np.array([0.485, 0.456, 0.406]))
    img = np.true_divide(img, np.array([0.229, 0.224, 0.225]))
    img = np.array(img, np.float32)
    img = img.transpose(2, 0, 1)
    img = img[None,:,:,:]

    return img, ori_img_size, output_size

def init_predictor(args):
    if args.model_dir is not "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()
    if args.use_gpu:
        config.enable_use_gpu(1000, 0)
    else:
        # If not specific mkldnn, you can set the blas thread.
        # The thread num should not be greater than the number of cores in the CPU.
        config.set_cpu_math_library_num_threads(4)
        config.enable_mkldnn()

    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="./inference.pdmodel",
        help="Model filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="./inference.pdiparams",
        help=
        "Parameter filename, Specify this when your model is a combined model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help=
        "Model dir, If you load a non-combined model, specify the directory of the model."
    )
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
    parser.add_argument("--use_gpu",
                        type=int,
                        default=0,
                        help="Whether use gpu.")
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    pred = init_predictor(args)
    K = np.array([[[2055.56, 0, 939.658], [0, 2055.56, 641.072], [0, 0, 1]]], np.float32)
    K_inverse = np.linalg.inv(K)

    img_path = args.input_path
    img, ori_img_size, output_size = get_img(img_path)
    ratio = get_ratio(ori_img_size, output_size)

    results = run(pred, [img, K_inverse, ratio])

    total_pred = paddle.to_tensor(results[0])

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
    
    
    img_draw = cv2.imread(img_path)
    for idx in range(bbox_3d.shape[0]):
        bbox = bbox_3d[idx]
        bbox = bbox.transpose([1,0]).numpy()
        img_draw = draw_box_3d(img_draw, bbox)
    
    cv2.imwrite(args.output_path, img_draw)

