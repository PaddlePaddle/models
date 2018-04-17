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

import paddle
import image_util
from paddle.utils.image_util import *
import random
from PIL import Image
from PIL import ImageDraw
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy
import functools


class Settings(object):
    def __init__(self, dataset, toy, data_dir, label_file, resize_h, resize_w,
                 mean_value, apply_distort, apply_expand):
        self._dataset = dataset
        self._toy = toy
        self._data_dir = data_dir
        if dataset == "pascalvoc":
            self._label_list = []
            label_fpath = os.path.join(data_dir, label_file)
            for line in open(label_fpath):
                self._label_list.append(line.strip())

        self._thread = 2
        self._buf_size = 2048
        self._apply_distort = apply_distort
        self._apply_expand = apply_expand
        self._resize_height = resize_h
        self._resize_width = resize_w
        self._img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')
        self._expand_prob = 0.5
        self._expand_max_ratio = 4
        self._hue_prob = 0.5
        self._hue_delta = 18
        self._contrast_prob = 0.5
        self._contrast_delta = 0.5
        self._saturation_prob = 0.5
        self._saturation_delta = 0.5
        self._brightness_prob = 0.5
        self._brightness_delta = 0.125

    @property
    def dataset(self):
        return self._dataset

    @property
    def toy(self):
        return self._toy

    @property
    def apply_distort(self):
        return self._apply_expand

    @property
    def apply_distort(self):
        return self._apply_distort

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        self._data_dir = data_dir

    @property
    def label_list(self):
        return self._label_list

    @property
    def resize_h(self):
        return self._resize_height

    @property
    def resize_w(self):
        return self._resize_width

    @property
    def img_mean(self):
        return self._img_mean


def process_image(sample, settings, mode):
    img = Image.open(sample[0])
    if img.mode == 'L':
        img = img.convert('RGB')
    img_width, img_height = img.size

    if mode == 'train' or mode == 'test':
        if settings.dataset == 'coco':
            # layout: category_id | xmin | ymin | xmax | ymax | iscrowd | origin_coco_bbox | segmentation | area | image_id | annotation_id
            bbox_labels = []
            annIds = coco.getAnnIds(imgIds=image['id'])
            anns = coco.loadAnns(annIds)
            for ann in anns:
                bbox_sample = []
                # start from 1, leave 0 to background
                bbox_sample.append(
                    float(category_ids.index(ann['category_id'])) + 1)
                bbox = ann['bbox']
                xmin, ymin, w, h = bbox
                xmax = xmin + w
                ymax = ymin + h
                bbox_sample.append(float(xmin) / img_width)
                bbox_sample.append(float(ymin) / img_height)
                bbox_sample.append(float(xmax) / img_width)
                bbox_sample.append(float(ymax) / img_height)
                bbox_sample.append(float(ann['iscrowd']))
                #bbox_sample.append(ann['bbox'])
                #bbox_sample.append(ann['segmentation'])
                #bbox_sample.append(ann['area'])
                #bbox_sample.append(ann['image_id'])
                #bbox_sample.append(ann['id'])
                bbox_labels.append(bbox_sample)
        elif settings.dataset == 'pascalvoc':
            # layout: label | xmin | ymin | xmax | ymax | difficult
            bbox_labels = []
            root = xml.etree.ElementTree.parse(sample[1]).getroot()
            for object in root.findall('object'):
                bbox_sample = []
                # start from 1
                bbox_sample.append(
                    float(settings.label_list.index(object.find('name').text)))
                bbox = object.find('bndbox')
                difficult = float(object.find('difficult').text)
                bbox_sample.append(float(bbox.find('xmin').text) / img_width)
                bbox_sample.append(float(bbox.find('ymin').text) / img_height)
                bbox_sample.append(float(bbox.find('xmax').text) / img_width)
                bbox_sample.append(float(bbox.find('ymax').text) / img_height)
                bbox_sample.append(difficult)
                bbox_labels.append(bbox_sample)

        sample_labels = bbox_labels
        if mode == 'train':
            if settings._apply_distort:
                img = image_util.distort_image(img, settings)
            if settings._apply_expand:
                img, bbox_labels, img_width, img_height = image_util.expand_image(
                    img, bbox_labels, img_width, img_height, settings)
            batch_sampler = []
            # hard-code here
            batch_sampler.append(
                image_util.sampler(1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0))
            batch_sampler.append(
                image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0))
            """ random crop """
            sampled_bbox = image_util.generate_batch_samples(batch_sampler,
                                                             bbox_labels)

            img = np.array(img)
            if len(sampled_bbox) > 0:
                idx = int(random.uniform(0, len(sampled_bbox)))
                img, sample_labels = image_util.crop_image(
                    img, bbox_labels, sampled_bbox[idx], img_width, img_height)

            img = Image.fromarray(img)
    img = img.resize((settings.resize_w, settings.resize_h), Image.ANTIALIAS)
    img = np.array(img)

    if mode == 'train':
        mirror = int(random.uniform(0, 2))
        if mirror == 1:
            img = img[:, ::-1, :]
            for i in xrange(len(sample_labels)):
                tmp = sample_labels[i][1]
                sample_labels[i][1] = 1 - sample_labels[i][3]
                sample_labels[i][3] = 1 - tmp

    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # RBG to BGR
    img = img[[2, 1, 0], :, :]
    img = img.astype('float32')
    img -= settings.img_mean
    img = img.flatten()
    img = img * 0.007843

    sample_labels = np.array(sample_labels)
    if mode == 'train' or mode == 'test':
        if len(sample_labels) != 0:
            return img.astype(
                'float32'), sample_labels[:, 1:5], sample_labels[:, 0].astype(
                    'int32'), sample_labels[:, -1].astype('int32')
    elif mode == 'infer':
        return img.astype('float32')


def _reader_creator(settings, file_list, mode, shuffle):
    def reader():
        if settings.dataset == 'coco':
            # cocoapi 
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            coco = COCO(file_list)
            image_ids = coco.getImgIds()
            images = coco.loadImgs(image_ids)
            category_ids = coco.getCatIds()
            category_names = [
                item['name'] for item in coco.loadCats(category_ids)
            ]
        elif settings.dataset == 'pascalvoc':
            flist = open(file_list)
            images = [line.strip() for line in flist]

        if not settings.toy == 0:
            images = images[:settings.toy] if len(
                images) > settings.toy else images
        print("{} on {} with {} images".format(mode, settings.dataset,
                                               len(images)))
        if shuffle:
            random.shuffle(images)

        for image in images:
            if settings.dataset == 'coco':
                image_name = image['file_name']
                image_path = os.path.join(settings.data_dir, image_name)
                yield [image_path]
            elif settings.dataset == 'pascalvoc':
                if mode == 'train' or mode == 'test':
                    image_path, label_path = image.split()
                    image_path = os.path.join(settings.data_dir, image_path)
                    label_path = os.path.join(settings.data_dir, label_path)
                    yield image_path, label_path
                elif mode == 'infer':
                    image_path = os.path.join(settings.data_dir, image)
                    yield [image_path]

    mapper = functools.partial(process_image, mode=mode, settings=settings)
    return paddle.reader.xmap_readers(mapper, reader, settings._thread,
                                      settings._buf_size)


def draw_bounding_box_on_image(image,
                               sample_labels,
                               image_name,
                               category_names,
                               color='red',
                               thickness=4,
                               with_text=True,
                               normalized=True):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if not normalized:
        im_width, im_height = 1, 1
    for item in sample_labels:
        label = item[0]
        category_name = category_names[int(label)]
        bbox = item[1:5]
        xmin, ymin, xmax, ymax = bbox
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line(
            [(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)
        if with_text:
            if image.mode == 'RGB':
                draw.text((left, top), category_name, (255, 255, 0))
    image.save(image_name)


def train(settings, file_list, shuffle=True):
    file_list = os.path.join(settings.data_dir, file_list)
    if settings.dataset == 'coco':
        train_settings = copy.copy(settings)
        if '2014' in file_list:
            sub_dir = "train2014"
        elif '2017' in file_list:
            sub_dir = "train2017"
        train_settings.data_dir = os.path.join(settings.data_dir, sub_dir)
        return _reader_creator(train_settings, file_list, 'train', shuffle)
    elif settings.dataset == 'pascalvoc':
        return _reader_creator(settings, file_list, 'train', shuffle)


def test(settings, file_list):
    file_list = os.path.join(settings.data_dir, file_list)
    if settings.dataset == 'coco':
        test_settings = copy.copy(settings)
        if '2014' in file_list:
            sub_dir = "val2014"
        elif '2017' in file_list:
            sub_dir = "val2017"
        test_settings.data_dir = os.path.join(settings.data_dir, sub_dir)
        return _reader_creator(test_settings, file_list, 'test', False)
    elif settings.dataset == 'pascalvoc':
        return _reader_creator(settings, file_list, 'test', False)


def infer(settings, file_list):
    return _reader_creator(settings, file_list, 'infer', False)
