#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import argparse
import numpy as np
import cv2

from paddle import fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def resize_short(img, target_size, interpolation=None):
    """resize image
    
    Args:
        img: image data
        target_size: resize short target size
        interpolation: interpolation mode

    Returns:
        resized image data
    """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    if interpolation:
        resized = cv2.resize(
            img, (resized_width, resized_height), interpolation=interpolation)
    else:
        resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center):
    """crop image 
    
    Args:
        img: images data
        target_size: crop target size
        center: crop mode
    
    Returns:
        img: cropped image data
    """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def preprocess_image(img_path):
    """ preprocess_image """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_size = 224
    target_size = 256

    img = cv2.imread(img_path)
    img = resize_short(img, target_size, interpolation=None)
    img = crop_image(img, target_size=crop_size, center=True)
    img = img[:, :, ::-1]

    img = img.astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    img = np.expand_dims(img, axis=0).copy()
    return img


def main():
    args = parse_args()

    # config AnalysisConfig
    config = AnalysisConfig(args.model_file, args.params_file)
    config.enable_use_gpu(1000)

    # create predictor
    predictor = create_paddle_predictor(config.to_native_config())

    # input
    inputs = preprocess_image(args.image_path)
    inputs = PaddleTensor(inputs)

    # predict
    outputs = predictor.run([inputs])

    # get output
    output = outputs[0]
    output = output.as_ndarray().flatten()

    cls = np.argmax(output)
    score = output[cls]
    print("class: ", cls)
    print("score: ", score)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", type=str, default="", help="model filename")
    parser.add_argument(
        "--params_file", type=str, default="", help="parameter filename")
    parser.add_argument("--image_path", type=str, default="", help="image_path")
    return parser.parse_args()


if __name__ == "__main__":
    main()
