# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


import random
import paddle
import numpy as np

from PIL import Image

from utils.voc import VOC

__all__ = ['voc_train', 'voc_val', 'voc_train_val', 'voc_test']

#  globals
data_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
data_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def mapper_train(sample):
    image_path, label_path, voc = sample
    image = Image.open(image_path, mode='r').convert('RGB')
    label = Image.open(label_path, mode='r')

    image, label = voc.sync_transform(image, label) 
    image_array = np.array(image)  # HWC
    label_array = np.array(label)  # HW

    image_array = image_array.transpose((2, 0, 1))  # CHW
    image_array = image_array / 255.0  
    image_array = (image_array - data_mean) / data_std  
    image_array = image_array.astype('float32')
    label_array = label_array.astype('int64')
    return image_array, label_array


def mapper_val(sample):
    image_path, label_path, city = sample
    image = Image.open(image_path, mode='r').convert('RGB')
    label = Image.open(label_path, mode='r')

    image, label = city.sync_val_transform(image, label)  
    image_array = np.array(image) 
    label_array = np.array(label)  

    image_array = image_array.transpose((2, 0, 1))  
    image_array = image_array / 255.0 
    image_array = (image_array - data_mean) / data_std  
    image_array = image_array.astype('float32')
    label_array = label_array.astype('int64')
    return image_array, label_array


def mapper_test(sample):
    image_path, label_path = sample  # label is path
    image = Image.open(image_path, mode='r').convert('RGB')
    image_array = image
    return image_array, label_path  # label is path


# 已完成， 引用时记得传入参数，root, base_size, crop_size等， gpu_num必须设置，否则syncBN会出现某些卡没有数据的情况
def voc_train(data_root='../dataset', base_size=768, crop_size=576, scale=True, xmap=True, batch_size=1, gpu_num=1):
    voc = VOC(root=data_root, split='train', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = voc.get_path_pairs()

    def reader():
        if len(image_path) % (batch_size * gpu_num) != 0:
            length = (len(image_path) // (batch_size * gpu_num)) * (batch_size * gpu_num)
        else:
            length = len(image_path)
        for i in range(length):
            if i == 0: 
                cc = list(zip(image_path, label_path))
                random.shuffle(cc)
                image_path[:], label_path[:] = zip(*cc)
            yield image_path[i], label_path[i], voc
    if xmap:
        return paddle.reader.xmap_readers(mapper_train, reader, 4, 32)
    else:
        return paddle.reader.map_readers(mapper_train, reader)


def voc_val(data_root='../dataset', base_size=768, crop_size=576, scale=True, xmap=True):
    voc = VOC(root=data_root, split='val', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = voc.get_path_pairs()

    def reader():
        for i in range(len(image_path)):
            yield image_path[i], label_path[i], voc

    if xmap:
        return paddle.reader.xmap_readers(mapper_val, reader, 4, 32)
    else:
        return paddle.reader.map_readers(mapper_val, reader)


def voc_train_val(data_root='./dataset', base_size=768, crop_size=576, scale=True, xmap=True, batch_size=1, gpu_num=1):
    voc = VOC(root=data_root, split='train_val', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = voc.get_path_pairs()

    def reader():
        if len(image_path) % (batch_size * gpu_num) != 0:
            length = (len(image_path) // (batch_size * gpu_num)) * (batch_size * gpu_num)
        else:
            length = len(image_path)
        for i in range(length):
            if i == 0: 
                cc = list(zip(image_path, label_path))
                random.shuffle(cc)
                image_path[:], label_path[:] = zip(*cc)
            yield image_path[i], label_path[i]

    if xmap:
        return paddle.reader.xmap_readers(mapper_train, reader, 4, 32)
    else:
        return paddle.reader.map_readers(mapper_train, reader)


def voc_test(split='test', base_size=2048, crop_size=1024, scale=True, xmap=True):
    # 实际未使用base_size, crop_size, scale
    voc = VOC(split=split, base_size=base_size, crop_size=crop_size, scale=scale)
    image_path = voc.get_path_pairs()

    def reader():
        for i in range(len(image_path[:1])):
            yield image_path[i], image_path[i]
    if xmap:
        return paddle.reader.xmap_readers(mapper_test, reader, 4, 32)
    else:
        return paddle.reader.map_readers(mapper_test, reader)
