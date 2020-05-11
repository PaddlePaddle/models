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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import functools
import numpy as np
import paddle
from imgtool import process_image
import paddle.fluid as fluid

random.seed(0)

DATA_DIR = "./data/Stanford_Online_Products/"
TRAIN_LIST = './data/Stanford_Online_Products/Ebay_train.txt'
VAL_LIST = './data/Stanford_Online_Products/Ebay_test.txt'


def init_sop(mode):
    if mode == 'train':
        train_data = {}
        train_image_list = []
        train_list = open(TRAIN_LIST, "r").readlines()
        for i, item in enumerate(train_list):
            items = item.strip().split()
            if items[0] == 'image_id':
                continue
            path = items[3]
            label = int(items[1]) - 1
            train_image_list.append((path, label))
            if label not in train_data:
                train_data[label] = []
            train_data[label].append(path)
        random.shuffle(train_image_list)
        print("{} dataset size: {}".format(mode, len(train_data)))
        return train_data, train_image_list
    else:
        val_data = {}
        val_image_list = []
        test_image_list = []
        val_list = open(VAL_LIST, "r").readlines()
        for i, item in enumerate(val_list):
            items = item.strip().split()
            if items[0] == 'image_id':
                continue
            path = items[3]
            label = int(items[1])
            val_image_list.append((path, label))
            test_image_list.append(path)
            if label not in val_data:
                val_data[label] = []
            val_data[label].append(path)
        print("{} dataset size: {}".format(mode, len(val_data)))
        if mode == 'val':
            return val_data, val_image_list
        else:
            return test_image_list

def common_iterator(data, settings):
    batch_size = settings.train_batch_size
    samples_each_class = settings.samples_each_class
    assert (batch_size % samples_each_class == 0)
    class_num = batch_size // samples_each_class 
    def train_iterator():
        count = 0
        labs = list(data.keys())
        lab_num = len(labs)
        ind = list(range(0, lab_num))
        while True:
            random.shuffle(ind)
            ind_sample = ind[:class_num]
            for ind_i in ind_sample:
                lab = labs[ind_i]
                data_list = data[lab]
                data_ind = list(range(0, len(data_list)))
                random.shuffle(data_ind)
                anchor_ind = data_ind[:samples_each_class]

                for anchor_ind_i in anchor_ind:
                    anchor_path = DATA_DIR + data_list[anchor_ind_i]
                    yield anchor_path, lab
            count += 1
            if count >= settings.total_iter_num + 1:
                return

    return train_iterator

def triplet_iterator(data, settings):
    batch_size = settings.train_batch_size
    assert (batch_size % 3 == 0)
    def train_iterator():
        total_count = settings.train_batch_size * (settings.total_iter_num + 1)
        count = 0
        labs = list(data.keys())
        lab_num = len(labs)
        ind = list(range(0, lab_num))
        while True:
            random.shuffle(ind)
            ind_pos, ind_neg = ind[:2]
            lab_pos = labs[ind_pos]
            pos_data_list = data[lab_pos]
            data_ind = list(range(0, len(pos_data_list)))
            random.shuffle(data_ind)
            anchor_ind, pos_ind = data_ind[:2]

            lab_neg = labs[ind_neg]
            neg_data_list = data[lab_neg]
            neg_ind = random.randint(0, len(neg_data_list) - 1)
            
            anchor_path = DATA_DIR + pos_data_list[anchor_ind]
            yield anchor_path, lab_pos
            pos_path = DATA_DIR + pos_data_list[pos_ind]
            yield pos_path, lab_pos
            neg_path = DATA_DIR + neg_data_list[neg_ind]
            yield neg_path, lab_neg
            count += 3
            if count >= total_count:
                return

    return train_iterator

def arcmargin_iterator(data, settings):
    def train_iterator():
        total_count = settings.train_batch_size * (settings.total_iter_num + 1)
        count = 0
        while True:
            for items in data:
                path, label = items
                path = DATA_DIR + path
                yield path, label
                count += 1
                if count >= total_count:
                    return
    return train_iterator

def image_iterator(data, mode):
    def val_iterator():
        for items in data:
            path, label = items
            path = DATA_DIR + path 
            yield path, label
    def test_iterator():
        for item in data:
            path = item
            path = DATA_DIR + path 
            yield [path]
    if mode == 'val':
        return val_iterator
    else:
        return test_iterator

def createreader(settings, mode):
    def metric_reader():
        if mode == 'train':
            train_data, train_image_list = init_sop('train')
            loss_name = settings.loss_name
            if loss_name in ["softmax", "arcmargin"]:
                return arcmargin_iterator(train_image_list, settings)()
            elif loss_name == 'triplet':
                return triplet_iterator(train_data, settings)()
            else:
                return common_iterator(train_data, settings)()
        elif mode == 'val':
            val_data, val_image_list = init_sop('val')
            return image_iterator(val_image_list, 'val')()
        else:
            test_image_list = init_sop('test')
            return image_iterator(test_image_list, 'test')()

    image_shape = settings.image_shape.split(',')
    assert(image_shape[1] == image_shape[2])
    image_size = int(image_shape[2])
    keep_order = False if mode != 'train' or settings.loss_name in ['softmax', 'arcmargin'] else True
    image_mapper = functools.partial(process_image,
            mode=mode, color_jitter=False, rotate=False, crop_size=image_size)
    reader = fluid.io.xmap_readers(
            image_mapper, metric_reader, 8, 1000, order=keep_order)
    return reader


def train(settings): 
    return createreader(settings, "train")

def test(settings):
    return createreader(settings, "val")

def infer(settings):
    return createreader(settings, "test")
