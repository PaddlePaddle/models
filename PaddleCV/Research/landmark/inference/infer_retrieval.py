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
import sys
sys.path.append('./so')
import time

import cv2
import numpy as np

from ConfigParser import ConfigParser
from PyCNNPredict import PyCNNPredict

def normwidth(size, margin=32):
    outsize = size // margin * margin
    outsize = max(outsize, margin)
    return outsize

def loadconfig(configurefile):
    "load config from file"
    config = ConfigParser()
    config.readfp(open(configurefile, 'r'))
    return config

def resize_short(img, target_size):
    """ resize_short """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))

    resized_width = normwidth(resized_width)
    resized_height = normwidth(resized_height)
    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
    return resized

def crop_image(img, target_size, center):
    """ crop_image """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def preprocessor(img, crop_size):
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

    h, w = img.shape[:2]
    ratio = float(max(w, h)) / min(w, h)
    if ratio > 3:
        crop_size = int(crop_size * 3 / ratio)
    img = resize_short(img, crop_size)

    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(img_mean).reshape((3, 1, 1))
    img_std = np.array(img_std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return img

def cosinedist(a, b):
    return np.dot(a, b) / (np.sum(a * a) * np.sum(b * b))**0.5

def test_retrieval(model_name):
    conf_file = './conf/paddle-retrieval.conf'
    prefix = model_name + "_"
    config = loadconfig(conf_file)
    predictor = PyCNNPredict()
    predictor.init(conf_file, prefix)
    input_size = config.getint(prefix + 'predict', 'input_size')

    img_names = [
        './test_data/0.jpg', 
        './test_data/1.jpg', 
        './test_data/2.jpg', 
        './test_data/3.jpg'
    ]
    img_feas = []
    for img_path in img_names:
        im = cv2.imread(img_path)
        if im is None:
            return None
        im = preprocessor(im, input_size)
        im_data_shape = np.array([1, im.shape[0], im.shape[1], im.shape[2]])
        im = im.flatten().astype(np.float32)
        inputdatas = [im]
        inputshapes = [im_data_shape.astype(np.int32)]
        run_time = 0
        starttime = time.time()
        res = predictor.predict(inputdatas, inputshapes, [])
        run_time += (time.time() - starttime)
        fea = res[0][0]
        img_feas.append(fea)
        print("Time:", run_time)

    for i in xrange(len(img_names)-1):
        cosdist = cosinedist(img_feas[0], img_feas[i+1])
        cosdist = max(min(cosdist, 1), 0)
        print('cosine dist between {} and {}: {}'.format(0, i+1, cosdist))

if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr,'tools.py command'
