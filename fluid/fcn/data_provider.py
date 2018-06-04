import os
import cv2
import random
import numpy as np


class Settings(object):
    def __init__(self, data_dir, resize_h, resize_w, mean_value):
        self._data_dir = data_dir
        self._resize_height = resize_h
        self._resize_width = resize_w
        self._img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def resize_h(self):
        return self._resize_height

    @property
    def resize_w(self):
        return self._resize_width

    @property
    def img_mean(self):
        return self._img_mean


def _reader_creator(settings, file_list, mode, shuffle):
    def reader():
        with open(file_list) as flist:
            lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                if mode == 'train' or mode == 'test':
                    img_path, label_path = line.split()
                    img_path = os.path.join(settings.data_dir, img_path)
                    label_path = os.path.join(settings.data_dir, label_path)

                    img_label = cv2.imread(label_path)
                    img_label = cv2.resize(
                        img_label, (settings.resize_w, settings.resize_h),
                        interpolation=cv2.INTER_NEAREST)
                elif mode == 'infer':
                    img_path = os.path.join(settings.data_dir, line)

                img = cv2.imread(img_path)
                img = cv2.resize(img, (settings.resize_w, settings.resize_h))

                img = np.transpose(img, (2, 0, 1))
                img = img.astype('float32')
                img -= settings.img_mean

                if mode == 'train' or mode == 'test':
                    img_label = img_label[:, :, 0]
                    yield img.astype('float32'), img_label.astype('int64')
                elif mode == 'infer':
                    yield img.astype('float32'), img_path

    return reader


def train(settings, file_list, shuffle=True):
    return _reader_creator(settings, file_list, 'train', shuffle)


def test(settings, file_list):
    return _reader_creator(settings, file_list, 'test', False)


def infer(settings, file_list):
    return _reader_creator(settings, file_list, 'infer', False)


if __name__ == '__main__':
    data_args = Settings(
        data_dir='./data',
        resize_h=300,
        resize_w=300,
        mean_value=[104, 117, 124])

    train_file_list = './data/voc2012_seg_train.txt'
    train(data_args, train_file_list)
