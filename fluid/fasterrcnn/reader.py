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


class Settings(object):
    def __init__(
            self,
            dataset=None,
            data_dir=None,
            label_file=None,
            resize_h=300,
            resize_w=300,
            mean_value=[127.77, 115.95, 102.98], ):
        self.dataset = dataset
        self.data_dir = data_dir
        self.label_list = []
        label_fpath = os.path.join(data_dir, label_file)
        for line in open(label_fpath):
            self.label_list.append(line.strip())

        self.resize_h = resize_h
        self.resize_w = resize_w
        self.img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')


def preprocess(img, bbox_labels, mode, settings):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    img = img.resize((settings.resize_w, settings.resize_h), Image.ANTIALIAS)
    img = np.array(img)

    if mode == 'train':
        flip = int(np.random.uniform())
        if flip == 1:
            img = img[:, :, ::-1]
            for i in six.moves.xrange(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = img_height - sampled_labels[i][3]
                sampled_labels[i][3] = img_height - tmp
    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img.astype('float32')
    img -= settings.img_mean
    return img, sampled_labels


def coco(settings, file_list, mode, shuffle):
    # cocoapi
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    print('loading coco...')
    coco = COCO(file_list)
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)
    category_ids = coco.getCatIds()
    category_names = [item['name'] for item in coco.loadCats(category_ids)]

    print("{} on {} with {} images".format(mode, settings.dataset, len(images)))

    def reader():
        if mode == "train" and shuffle:
            random.shuffle(images)
        for image in images:
            image_name = image['file_name']
            image_path = os.path.join(settings.data_dir, image_name)

            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size
            im_id = image['id']
            im_info = [float(im_width), float(im_height), 16.0]
            # layout: category_id | xmin | ymin | xmax | ymax | im_info
            bbox_labels = []
            annIds = coco.getAnnIds(imgIds=image['id'])
            anns = coco.loadAnns(annIds)
            for ann in anns:
                bbox_sample = []
                # start from 1, leave 0 to background
                bbox_sample.append(float(ann['category_id']))
                #float(category_ids.index(ann['category_id'])) + 1)
                bbox = ann['bbox']
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                bbox_sample.append(float(xmin))
                bbox_sample.append(float(ymin))
                bbox_sample.append(float(xmax))
                bbox_sample.append(float(ymax))
                bbox_labels.append(bbox_sample)
            im, sample_labels = preprocess(im, bbox_labels, mode, settings)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue
            im = im.astype('float32')
            boxes = sample_labels[:, 1:5]
            lbls = sample_labels[:, 0].astype('int32')
            yield im, boxes, lbls, im_info


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
            label_path = os.path.join(settings.data_dir, label_path)

            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size
            im_info = [float(im_width), float(im_height), 16.0]
            # layout: label | xmin | ymin | xmax | ymax | im_info
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
            im, sample_labels = preprocess(im, bbox_labels, mode, settings)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue
            im = im.astype('float32')
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
