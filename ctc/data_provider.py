from __future__ import absolute_import
from __future__ import division

import os
from paddle.v2.image import load_image
import cv2


class AsciiDic(object):
    UNK = 0

    def __init__(self):
        self.dic = {
            '<unk>': self.UNK,
        }
        self.chars = [chr(i) for i in range(40, 171)]
        for id, c in enumerate(self.chars):
            self.dic[c] = id + 1

    def lookup(self, w):
        return self.dic.get(w, self.UNK)

    def id2word(self):
        self.id2word = {}
        for key, value in self.dic.items():
            self.id2word[value] = key

        return self.id2word

    def word2ids(self, sent):
        '''
        transform a word to a list of ids.
        @sent: str
        '''
        return [self.lookup(c) for c in list(sent)]

    def size(self):
        return len(self.dic)


class ImageDataset(object):
    def __init__(self,
                 train_image_paths_generator,
                 test_image_paths_generator,
                 infer_image_paths_generator,
                 fixed_shape=None,
                 is_infer=False):
        '''
        @image_paths_generator: function
            return a list of images' paths, called like:

                for path in image_paths_generator():
                    load_image(path)
        '''
        if is_infer == False:
            self.train_filelist = [p for p in train_image_paths_generator]
            self.test_filelist = [p for p in test_image_paths_generator]
        else:
            self.infer_filelist = [p for p in infer_image_paths_generator]

        self.fixed_shape = fixed_shape
        self.ascii_dic = AsciiDic()

    def train(self):
        for i, (image, label) in enumerate(self.train_filelist):
            yield self.load_image(image), self.ascii_dic.word2ids(label)

    def test(self):
        for i, (image, label) in enumerate(self.test_filelist):
            yield self.load_image(image), self.ascii_dic.word2ids(label)

    def infer(self):
        for i, (image, label) in enumerate(self.infer_filelist):
            yield self.load_image(image), label

    def load_image(self, path):
        '''
        load image and transform to 1-dimention vector
        '''
        image = load_image(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize all images to a fixed shape

        if self.fixed_shape:
            image = cv2.resize(
                image, self.fixed_shape, interpolation=cv2.INTER_CUBIC)

        image = image.flatten() / 255.
        return image


def get_file_list(image_file_list):
    pwd = os.path.dirname(image_file_list)
    with open(image_file_list) as f:
        for line in f:
            fs = line.strip().split(',')
            file = fs[0].strip()
            path = os.path.join(pwd, file)
            yield path, fs[1][2:-1]
