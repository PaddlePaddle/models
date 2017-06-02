from __future__ import absolute_import
from __future__ import division

import os
from paddle.v2.image import load_image, to_chw
import cv2


class AsciiDic(object):
    UNK = 0

    def __init__(self):
        self.dic = {
            '<unk>': self.UNK,
        }
        self.chars = [chr(i) for i in range(40, 171)]
        for id, c in enumerate(self.chars):
            self.dic[c] = id

    def lookup(self, w):
        return self.dic.get(w, self.UNK)

    def word2ids(self, sent):
        '''
        transform a word to a list of ids.
        @sent: str
        '''
        return [self.lookup(c) for c in list(sent)]

    def size(self):
        return len(self.chars)


class ImageDataset(object):
    def __init__(self,
                 image_paths_generator,
                 fixed_shape=None,
                 testset_size=1000):
        '''
        @image_paths_generator: function
            return a list of images' paths, called like:

                for path in image_paths_generator():
                    load_image(path)
        '''
        self.filelist = image_paths_generator
        self.fixed_shape = fixed_shape
        self.testset_size = testset_size
        self.ascii_dic = AsciiDic()

    def train(self):
        for i, (image, label) in enumerate(self.filelist):
            if i > self.testset_size:
                record = self.load_image(image), self.ascii_dic.word2ids(label)
                print record[0].shape, record[1]
                yield record

    def test(self):
        for i, (image, label) in enumerate(self.filelist):
            if i < self.testset_size:
                yield self.load_image(image), self.ascii_dic.word2ids(label)

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
        # image = to_chw(image)
        image = image.flatten() / 255.
        return image


def get_file_list(image_file_list):
    pwd = os.path.dirname(image_file_list)
    with open(image_file_list) as f:
        for line in f:
            fs = line.strip().split(',')
            file = fs[0] + '.jpg'
            path = os.path.join(pwd, file)
            yield path, fs[1]


if __name__ == '__main__':
    image_file_list = '/home/disk1/yanchunwei/90kDICT32px/train_all.txt'

    image_dataset = ImageDataset(
        get_file_list(image_file_list), fixed_shape=(173, 46))

    for i, image in enumerate(image_dataset.train()):
        print 'image', image
        if i > 10:
            break
