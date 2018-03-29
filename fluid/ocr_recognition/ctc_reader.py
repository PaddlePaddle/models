import os
import cv2
import tarfile
import numpy as np
from PIL import Image
from os import path
from paddle.v2.image import load_image
import paddle.v2 as paddle

NUM_CLASSES = 10784
DATA_SHAPE = [1, 48, 512]

DATA_MD5 = "1de60d54d19632022144e4e58c2637b5"
DATA_URL = "http://cloud.dlnel.org/filepub/?uuid=df937251-3c0b-480d-9a7b-0080dfeee65c"
CACHE_DIR_NAME = "ctc_data"
SAVED_FILE_NAME = "data.tar.gz"
DATA_DIR_NAME = "data"
TRAIN_DATA_DIR_NAME = "train_images"
TEST_DATA_DIR_NAME = "test_images"
TRAIN_LIST_FILE_NAME = "train.list"
TEST_LIST_FILE_NAME = "test.list"


class DataGenerator(object):
    def __init__(self):
        pass

    def train_reader(self, img_root_dir, img_label_list, batchsize):
        '''
        Reader interface for training.

        :param img_root_dir: The root path of the image for training.
        :type file_list: str

        :param img_label_list: The path of the <image_name, label> file for training.
        :type file_list: str

        '''

        img_label_lines = []
        if batchsize == 1:
            to_file = "tmp.txt"
            cmd = "cat " + img_label_list + " | awk '{print $1,$2,$3,$4;}' | shuf > " + to_file
            print "cmd: " + cmd
            os.system(cmd)
            print "finish batch shuffle"
            img_label_lines = open(to_file, 'r').readlines()
        else:
            to_file = "tmp.txt"
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
            print "cmd: " + cmd
            os.system(cmd)
            print "finish batch shuffle"
            img_label_lines = open(to_file, 'r').readlines()

        def reader():
            sizes = len(img_label_lines) / batchsize
            for i in range(sizes):
                result = []
                sz = [0, 0]
                for j in range(batchsize):
                    line = img_label_lines[i * batchsize + j]
                    # h, w, img_name, labels
                    items = line.split(' ')

                    label = [int(c) for c in items[-1].split(',')]
                    img = Image.open(os.path.join(img_root_dir, items[
                        2])).convert('L')  #zhuanhuidu
                    if j == 0:
                        sz = img.size
                    img = img.resize((sz[0], sz[1]))
                    img = np.array(img) - 127.5
                    img = img[np.newaxis, ...]
                    result.append([img, label])
                yield result

        return reader

    def test_reader(self, img_root_dir, img_label_list):
        '''
        Reader interface for inference.

        :param img_root_dir: The root path of the images for training.
        :type file_list: str

        :param img_label_list: The path of the <image_name, label> file for testing.
        :type file_list: list
        '''

        def reader():
            for line in open(img_label_list):
                # h, w, img_name, labels
                items = line.split(' ')

                label = [int(c) for c in items[-1].split(',')]
                img = Image.open(os.path.join(img_root_dir, items[2])).convert(
                    'L')
                img = np.array(img) - 127.5
                img = img[np.newaxis, ...]
                yield img, label

        return reader


def num_classes():
    '''Get classes number of this dataset.
    '''
    return NUM_CLASSES


def data_shape():
    '''Get image shape of this dataset. It is a dummy shape for this dataset.
    '''
    return DATA_SHAPE


def train(batch_size):
    generator = DataGenerator()
    data_dir = download_data()
    return generator.train_reader(
        path.join(data_dir, TRAIN_DATA_DIR_NAME),
        path.join(data_dir, TRAIN_LIST_FILE_NAME), batch_size)


def test(batch_size=1):
    generator = DataGenerator()
    data_dir = download_data()
    return paddle.batch(
        generator.test_reader(
            path.join(data_dir, TRAIN_DATA_DIR_NAME),
            path.join(data_dir, TRAIN_LIST_FILE_NAME)), batch_size)


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
