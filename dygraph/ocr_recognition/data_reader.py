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

SOS = 0
EOS = 1
NUM_CLASSES = 95
DATA_SHAPE = [1, 48, 512]

DATA_MD5 = "7256b1d5420d8c3e74815196e58cdad5"
DATA_URL = "http://paddle-ocr-data.bj.bcebos.com/data.tar.gz"
CACHE_DIR_NAME = "attention_data"
SAVED_FILE_NAME = "data.tar.gz"
DATA_DIR_NAME = "data"
TRAIN_DATA_DIR_NAME = "train_images"
TEST_DATA_DIR_NAME = "test_images"
TRAIN_LIST_FILE_NAME = "train.list"
TEST_LIST_FILE_NAME = "test.list"


class DataGenerator(object):
    def __init__(self):
        pass

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
        if not shuffle:
            cmd = "cat " + img_label_list + " | awk '{print $1,$2,$3,$4;}' > " + to_file
        elif batchsize == 1:
            cmd = "cat " + img_label_list + " | awk '{print $1,$2,$3,$4;}' | shuf > " + to_file
        else:
            #cmd1: partial shuffle
            cmd = "cat " + img_label_list + " | awk '{printf(\"%04d%.4f %s\\n\", $1, rand(), $0)}' | sort | sed 1,$((1 + RANDOM % 100))d | "
            #cmd2: batch merge and shuffle
            cmd += "awk '{printf $2\" \"$3\" \"$4\" \"$5\" \"; if(NR % " + str(
                batchsize) + " == 0) print \"\";}' | shuf | "
            #cmd3: batch split
            cmd += "awk '{if(NF == " + str(
                batchsize
            ) + " * 4) {for(i = 0; i < " + str(
                batchsize
            ) + "; i++) print $(4*i+1)\" \"$(4*i+2)\" \"$(4*i+3)\" \"$(4*i+4);}}' > " + to_file
        os.system(cmd)
        print("finish batch shuffle")
        img_label_lines = open(to_file, 'r').readlines()

        def reader():
            sizes = len(img_label_lines) // batchsize
            if sizes == 0:
                raise ValueError('batchsize is bigger than the dataset size.')
            while True:
                for i in range(sizes):
                    result = []
                    sz = [0, 0]
                    max_len = 0
                    for k in range(batchsize):
                        line = img_label_lines[i * batchsize + k]
                        items = line.split(' ')
                        label = [int(c) for c in items[-1].split(',')]
                        max_len = max(max_len, len(label))

                    for j in range(batchsize):
                        line = img_label_lines[i * batchsize + j]
                        items = line.split(' ')
                        label = [int(c) for c in items[-1].split(',')]

                        mask = np.zeros((max_len)).astype('float32')
                        mask[:len(label) + 1] = 1.0
                        #mask[ j, :len(label) + 1] = 1.0
                        if max_len > len(label) + 1:
                            extend_label = [EOS] * (max_len - len(label) - 1)
                            label.extend(extend_label)
                        else:
                            label = label[0:max_len - 1]
                        img = Image.open(os.path.join(img_root_dir, items[
                            2])).convert('L')
                        if j == 0:
                            sz = img.size
                        img = img.resize((sz[0], sz[1]))
                        img = np.array(img) - 127.5
                        img = img[np.newaxis, ...]
                        result.append([img, [SOS] + label, label + [EOS], mask])
                    yield result
                if not cycle:
                    break

        return reader


def num_classes():
    '''Get classes number of this dataset.
    '''
    return NUM_CLASSES


def data_shape():
    '''Get image shape of this dataset. It is a dummy shape for this dataset.
    '''
    return DATA_SHAPE


def data_reader(batch_size,
                images_dir=None,
                list_file=None,
                cycle=False,
                shuffle=False,
                data_type="train"):
    generator = DataGenerator()

    if data_type == "train":
        if images_dir is None:
            data_dir = download_data()
            images_dir = path.join(data_dir, TRAIN_DATA_DIR_NAME)
        if list_file is None:
            list_file = path.join(data_dir, TRAIN_LIST_FILE_NAME)
    elif data_type == "test":
        if images_dir is None:
            data_dir = download_data()
            images_dir = path.join(data_dir, TEST_DATA_DIR_NAME)
        if list_file is None:
            list_file = path.join(data_dir, TEST_LIST_FILE_NAME)
    else:
        print("data type only support train | test")
        raise Exception("data type only support train | test")
    return generator.train_reader(
        images_dir, list_file, batch_size, cycle, shuffle=shuffle)


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
