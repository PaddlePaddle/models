import os
import cv2
import numpy as np

from paddle.v2.image import load_image


class DataGenerator(object):
    def __init__(self):
        pass

    def train_reader(self, img_root_dir, img_label_list):
        '''
        Reader interface for training.

		:param img_root_dir: The root path of the image for training.
        :type file_list: str 

        :param img_label_list: The path of the <image_name, label> file for training.
        :type file_list: str 

        '''
        # sort by height, e.g. idx
        img_label_lines = []
        for line in open(img_label_list):
            # h, w, img_name, labels
            items = line.split(' ')
            idx = "{:0>5d}".format(int(items[0]))
            img_label_lines.append(idx + ' ' + line)
        img_label_lines.sort()

        def reader():
            for line in img_label_lines:
                # h, w, img_name, labels
                items = line.split(' ')[1:]

                assert len(items) == 4

                label = [int(c) for c in items[-1].split(',')]

                img = load_image(os.path.join(img_root_dir, items[2]))
                img = np.transpose(img, (2, 0, 1))
                #img = img[np.newaxis, ...]

                yield img, label

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

                assert len(items) == 4

                label = [int(c) for c in items[-1].split(',')]

                img = load_image(os.path.join(img_root_dir, items[2]))
                img = np.transpose(img, (2, 0, 1))
                #img = img[np.newaxis, ...]

                yield img, label

        return reader
