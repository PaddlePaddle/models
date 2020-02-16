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
import cv2
import tarfile
import numpy as np
from PIL import Image
from os import path
from paddle.dataset.image import load_image
import paddle
import random

try:
    input = raw_input
except NameError:
    pass

SOS = 0
EOS = 1
NUM_CLASSES = 95
IMG_WIDTH = 384
DATA_SHAPE = [1, 48, IMG_WIDTH]

DATA_MD5 = "7256b1d5420d8c3e74815196e58cdad5"
DATA_URL = "http://paddle-ocr-data.bj.bcebos.com/data.tar.gz"
CACHE_DIR_NAME = "ctc_data"
SAVED_FILE_NAME = "data.tar.gz"
DATA_DIR_NAME = "data"
TRAIN_DATA_DIR_NAME = "train_images"
TEST_DATA_DIR_NAME = "test_images"
TRAIN_LIST_FILE_NAME = "train.list"
TEST_LIST_FILE_NAME = "test.list"


class DataGenerator(object):
    def __init__(self, model="crnn_ctc"):
        self.model = model

    def train_reader(self,
                     img_root_dir,
                     img_label_list,
                     batchsize,
                     cycle,
                     shuffle=True):
        '''
        Reader interface for training.

        :param img_root_dir: The root path of the image for training.
        :type img_root_dir: str

        :param img_label_list: The path of the <image_name, label> file for training.
        :type img_label_list: str

        :param cycle: If number of iterations is greater than dataset_size / batch_size
        it reiterates dataset over as many times as necessary.
        :type cycle: bool

        '''

        img_label_lines = []
        to_file = "tmp.txt"

        def _shuffle_data(input_file_path, output_file_path, shuffle,
                          batchsize):
            def _write_file(file_path, lines_to_write):
                open(file_path, 'w').writelines(
                    ["{}\n".format(item) for item in lines_to_write])

            input_file = open(input_file_path, 'r')
            lines_to_shuf = [line.strip() for line in input_file.readlines()]

            if not shuffle:
                _write_file(output_file_path, lines_to_shuf)
            elif batchsize == 1:
                random.shuffle(lines_to_shuf)
                _write_file(output_file_path, lines_to_shuf)
            else:
                #partial shuffle
                for i in range(len(lines_to_shuf)):
                    str_i = lines_to_shuf[i]
                    list_i = str_i.strip().split(' ')
                    str_i_ = "%04d%.4f " % (int(list_i[0]), random.random()
                                            ) + str_i
                    lines_to_shuf[i] = str_i_
                lines_to_shuf.sort()
                delete_num = random.randint(1, 100)
                del lines_to_shuf[0:delete_num]

                #batch merge and shuffle
                lines_concat = []
                for i in range(0, len(lines_to_shuf), batchsize):
                    lines_concat.append(' '.join(lines_to_shuf[i:i +
                                                               batchsize]))
                random.shuffle(lines_concat)

                #batch split
                out_file = open(output_file_path, 'w')
                for i in range(len(lines_concat)):
                    tmp_list = lines_concat[i].split(' ')
                    for j in range(int(len(tmp_list) / 5)):
                        out_file.write("{} {} {} {}\n".format(tmp_list[
                            5 * j + 1], tmp_list[5 * j + 2], tmp_list[
                                5 * j + 3], tmp_list[5 * j + 4]))
                out_file.close()
            input_file.close()

        _shuffle_data(img_label_list, to_file, shuffle, batchsize)
        print("finish batch shuffle")
        img_label_lines = open(to_file, 'r').readlines()

        def reader():
            sizes = len(img_label_lines) // batchsize
            if sizes == 0:
                raise ValueError('Batch size is bigger than the dataset size.')
            while True:
                for i in range(sizes):
                    result = []
                    sz = [0, 0]
                    for j in range(batchsize):
                        line = img_label_lines[i * batchsize + j]
                        # h, w, img_name, labels
                        items = line.split(' ')

                        label = [int(c) for c in items[-1].split(',')]
                        img = Image.open(os.path.join(img_root_dir, items[
                            2])).convert('L')
                        if j == 0:
                            sz = img.size
                        img = img.resize((sz[0], DATA_SHAPE[1]))
                        img = np.array(img) - 127.5
                        img = img[np.newaxis, ...]
                        if self.model == "crnn_ctc":
                            result.append([img, label])
                        else:
                            result.append([img, [SOS] + label, label + [EOS]])
                    yield result
                if not cycle:
                    break

        return reader

    def test_reader(self, img_root_dir, img_label_list):
        '''
        Reader interface for inference.

        :param img_root_dir: The root path of the images for training.
        :type img_root_dir: str

        :param img_label_list: The path of the <image_name, label> file for testing.
        :type img_label_list: str
        '''

        def reader():
            for line in open(img_label_list):
                # h, w, img_name, labels
                items = line.split(' ')

                label = [int(c) for c in items[-1].split(',')]
                img = Image.open(os.path.join(img_root_dir, items[2])).convert(
                    'L')

                img = img.resize((img.size[0], DATA_SHAPE[1])) # resize height
                img = np.array(img) - 127.5
                img = img[np.newaxis, ...]
                if self.model == "crnn_ctc":
                    yield img, label
                else:
                    yield img, [SOS] + label, label + [EOS]

        return reader

    def infer_reader(self, img_root_dir=None, img_label_list=None, cycle=False):
        '''A reader interface for inference.

        :param img_root_dir: The root path of the images for training.
        :type img_root_dir: str

        :param img_label_list: The path of the <image_name, label> file for
        inference. It should be the path of <image_path> file if img_root_dir
        was None. If img_label_list was set to None, it will read image path
        from stdin.
        :type img_root_dir: str

        :param cycle: If number of iterations is greater than dataset_size /
        batch_size it reiterates dataset over as many times as necessary.
        :type cycle: bool
        '''

        def reader():
            def yield_img_and_label(lines):
                for line in lines:
                    if img_root_dir is not None:
                        # h, w, img_name, labels
                        img_name = line.split(' ')[2]
                        img_path = os.path.join(img_root_dir, img_name)
                    else:
                        img_path = line.strip("\t\n\r")
                    img = Image.open(img_path).convert('L')
                    img = img.resize((img.size[0], DATA_SHAPE[1])) # resize height
                    img = np.array(img) - 127.5
                    img = img[np.newaxis, ...]
                    yield img, [[0]]

            if img_label_list is not None:
                lines = []
                with open(img_label_list) as f:
                    lines = f.readlines()
                for img, label in yield_img_and_label(lines):
                    yield img, label
                while cycle:
                    for img, label in yield_img_and_label(lines):
                        yield img, label
            else:
                while True:
                    img_path = input("Please input the path of image: ")
                    img = Image.open(img_path).convert('L')
                    img = img.resize((img.size[0], DATA_SHAPE[1])) # resize height
                    img = np.array(img) - 127.5
                    img = img[np.newaxis, ...]
                    yield img, [[0]]

        return reader


def num_classes():
    '''Get classes number of this dataset.
    '''
    return NUM_CLASSES


def data_shape():
    '''Get image shape of this dataset. It is a dummy shape for this dataset.
    '''
    return DATA_SHAPE


def train(batch_size,
          train_images_dir=None,
          train_list_file=None,
          cycle=False,
          model="crnn_ctc"):
    generator = DataGenerator(model)
    if train_images_dir is None:
        data_dir = download_data()
        train_images_dir = path.join(data_dir, TRAIN_DATA_DIR_NAME)
    if train_list_file is None:
        train_list_file = path.join(data_dir, TRAIN_LIST_FILE_NAME)
    shuffle = True
    if 'ce_mode' in os.environ:
        shuffle = False
    return generator.train_reader(
        train_images_dir, train_list_file, batch_size, cycle, shuffle=shuffle)


def test(batch_size=1,
         test_images_dir=None,
         test_list_file=None,
         model="crnn_ctc"):
    generator = DataGenerator(model)
    if test_images_dir is None:
        data_dir = download_data()
        test_images_dir = path.join(data_dir, TEST_DATA_DIR_NAME)
    if test_list_file is None:
        test_list_file = path.join(data_dir, TEST_LIST_FILE_NAME)
    return paddle.batch(
        generator.test_reader(test_images_dir, test_list_file), batch_size)


def inference(batch_size=1,
              infer_images_dir=None,
              infer_list_file=None,
              cycle=False,
              model="crnn_ctc"):
    generator = DataGenerator(model)
    return paddle.batch(
        generator.infer_reader(infer_images_dir, infer_list_file, cycle),
        batch_size)


def download_data():
    '''Download train and test data.
    '''
    tar_file = paddle.dataset.common.download(
        DATA_URL, CACHE_DIR_NAME, DATA_MD5, save_name=SAVED_FILE_NAME)
    data_dir = path.join(path.dirname(tar_file), DATA_DIR_NAME)
    if not path.isdir(data_dir):
        t = tarfile.open(tar_file, "r:gz")
        t.extractall(path=path.dirname(tar_file))
        t.close()
    return data_dir
