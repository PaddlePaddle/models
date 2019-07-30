from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from PIL import Image
import numpy as np

A_LIST_FILE = "./data/horse2zebra/trainA.txt"
B_LIST_FILE = "./data/horse2zebra/trainB.txt"
A_TEST_LIST_FILE = "./data/horse2zebra/testA.txt"
B_TEST_LIST_FILE = "./data/horse2zebra/testB.txt"
IMAGES_ROOT = "./data/horse2zebra/"


def image_shape():
    return [3, 256, 256]


def max_images_num():
    return 1335


def reader_creater(list_file, cycle=True, shuffle=True, return_name=False):
    images = [IMAGES_ROOT + line for line in open(list_file, 'r').readlines()]

    def reader():
        while True:
            if shuffle:
                np.random.shuffle(images)
            for file in images:
                file = file.strip("\n\r\t ")
                image = Image.open(file)
                image = image.resize((256, 256))
                image = np.array(image) / 127.5 - 1
                if len(image.shape) != 3:
                    continue
                image = image[:, :, 0:3].astype("float32")
                image = image.transpose([2, 0, 1])
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
