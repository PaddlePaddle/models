# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.utils.image_util import *
import random
from PIL import Image
from PIL import ImageDraw
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy
import six
import cv2


class Settings(object):
    def __init__(
            self,
            dataset=None,
            data_dir=None,
            label_file=None,
            resize_h=800,
            resize_w=1333,
            mean_value=np.array([[[102.98, 115.95, 127.77]]]), ):
        self.dataset = dataset
        self.data_dir = data_dir
        self.label_list = []
        label_fpath = os.path.join(data_dir, label_file)
        for line in open(label_fpath):
            self.label_list.append(line.strip())
        self.img_mean = mean_value
        self.resize_h = resize_h
        self.resize_w = resize_w


def preprocess(img, bbox_labels, mode, settings):
    img = img.astype(np.float32, copy=False)
    img -= settings.img_mean
    sampled_labels = bbox_labels
    img_shape = img.shape
    img_size_min = np.min(img_shape[0:2])
    img_size_max = np.max(img_shape[0:2])
    im_scale = float(settings.resize_h) / float(img_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * img_size_max) > settings.resize_w:
        im_scale = float(settings.resize_w) / float(img_size_max)
    img = cv2.resize(
        img,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR)
    channel_swap = (2, 0, 1)  #(batch, channel, height, width)
    img = img.transpose(channel_swap)
    return img, sampled_labels, im_scale


def pascalvoc(settings, file_list, mode, shuffle):
    flist = open(file_list)
    images = [line.strip() for line in flist]
    print("{} on {} with {} images".format(mode, settings.dataset, len(images)))

    def reader():
        if mode == "train" and shuffle:
            random.shuffle(images)
        for image in images:
            image_path, label_path = image.split()
            image_path = os.path.join(settings.data_dir, image_path)
            #if '005393.jpg' not in image_path: continue
            label_path = os.path.join(settings.data_dir, label_path)
            im = cv2.imread(image_path)
            im_width = im.shape[0]
            im_height = im.shape[1]
            #im = Image.open(image_path)
            #if im.mode == 'L':
            #    im = im.convert('RGB')
            #im_width, im_height = im.size
            # layout: label | xmin | ymin | xmax | ymax | im_info
            print(image_path)
            bbox_labels = []
            root = xml.etree.ElementTree.parse(label_path).getroot()
            for object in root.findall('object'):
                bbox_sample = []
                # start from 1
                bbox_sample.append(
                    float(settings.label_list.index(object.find('name').text)))
                bbox = object.find('bndbox')
                bbox_sample.append(float(bbox.find('xmin').text))
                bbox_sample.append(float(bbox.find('ymin').text))
                bbox_sample.append(float(bbox.find('xmax').text))
                bbox_sample.append(float(bbox.find('ymax').text))
                bbox_labels.append(bbox_sample)
            im, sample_labels, im_scale = preprocess(im, bbox_labels, mode,
                                                     settings)
            sample_labels = np.array(sample_labels)
            im_info = [
                float(im_width) * im_scale, float(im_height) * im_scale,
                im_scale
            ]
            if len(sample_labels) == 0: continue
            boxes = sample_labels[:, 1:5]
            lbls = sample_labels[:, 0].astype('int32')
            is_crowd = np.zeros(len(boxes), dtype='int32')
            yield im, boxes, lbls, is_crowd, im_info

    return reader


def train(settings, file_list, shuffle=True):
    file_list = os.path.join(settings.data_dir, file_list)
    return pascalvoc(settings, file_list, 'train', shuffle)


def test(settings, file_list):
    file_list = os.path.join(settings.data_dir, file_list)
    return pascalvoc(settings, file_list, 'test', False)


def infer(settings, image_path):
    def reader():
        img = Image.open(image_path)
        if img.mode == 'L':
            img = im.convert('RGB')
        im_width, im_height = img.size
        img = img.resize((settings.resize_w, settings.resize_h),
                         Image.ANTIALIAS)
        img = np.array(img)
        # HWC to CHW
        if len(img.shape) == 3:
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 1, 0)
        img -= settings.img_mean
        return img

    return reader
