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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from PIL import Image, ImageOps
import numpy as np

DATASET = "cityscapes"
A_LIST_FILE = "./data/"+DATASET+"/trainA.txt"
B_LIST_FILE = "./data/"+DATASET+"/trainB.txt"
A_TEST_LIST_FILE = "./data/"+DATASET+"/testA.txt"
B_TEST_LIST_FILE = "./data/"+DATASET+"/testB.txt"
IMAGES_ROOT = "./data/"+DATASET+"/"

def image_shape():
    return [3, 256, 256]


def max_images_num():
    return 2974


def reader_creater(list_file, cycle=True, shuffle=True, return_name=False):
    images = [IMAGES_ROOT + line for line in open(list_file, 'r').readlines()]

    def reader():
        while True:
            if shuffle:
                np.random.shuffle(images)
            for file in images:
                file = file.strip("\n\r\t ")
                image = Image.open(file)
                ## Resize
                image = image.resize((286, 286), Image.BICUBIC)
                ## RandomCrop
                i = np.random.randint(0, 30)
                j = np.random.randint(0, 30)
                image = image.crop((i, j , i+256, j+256))
                # RandomHorizontalFlip
                sed = np.random.rand()
                if sed > 0.5:
                    image = ImageOps.mirror(image)
                # ToTensor
                image = np.array(image).transpose([2, 0, 1]).astype('float32')
                image = image / 255.0
                # Normalize, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
                image = (image - 0.5) / 0.5
                
                if return_name:
                    yield image[np.newaxis, :], os.path.basename(file)
                else:
                    yield image
            if not cycle:
                break

    return reader


def a_reader(shuffle=True):
    """
    Reader of images with A style for training.
    """
    return reader_creater(A_LIST_FILE, shuffle=shuffle)


def b_reader(shuffle=True):
    """
    Reader of images with B style for training.
    """
    return reader_creater(B_LIST_FILE, shuffle=shuffle)


def a_test_reader():
    """
    Reader of images with A style for test.
    """
    return reader_creater(A_TEST_LIST_FILE, cycle=False, return_name=True)


def b_test_reader():
    """
    Reader of images with B style for test.
    """
    return reader_creater(B_TEST_LIST_FILE, cycle=False, return_name=True)
