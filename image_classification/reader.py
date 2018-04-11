import random
from paddle.v2.image import load_and_transform
import paddle.v2 as paddle
from multiprocessing import cpu_count


def train_mapper(sample):
    '''
    map image path to type needed by model input layer for the training set
    '''
    img, label = sample
    img = paddle.image.load_image(img)
    # https://github.com/PaddlePaddle/Paddle/blob/d52fa26fdab7a0497a3e7f49833d1b3827955c44/python/paddle/v2/dataset/flowers.py#L65
    # mean:根据加载的预训练模型的mean设置。如果不加载预训练模型，则根据自己喜好设置
    # paddlepaddle的代码中，默认减去这个mean值，如果加载的预训练模型使用的mean与这个值不一样，需要修改
    img = paddle.image.simple_transform(img, 256, 224, True, mean=[103.94, 116.78, 123.68])
    return img.flatten().astype('float32'), label


def test_mapper(sample):
    '''
    map image path to type needed by model input layer for the test set
    '''
    img, label = sample
    img = paddle.image.load_image(img)
    # 同上
    img = paddle.image.simple_transform(img, 256, 224, True, mean=[103.94, 116.78, 123.68])
    return img.flatten().astype('float32'), label


def train_reader(train_list, buffered_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper, reader,
                                      cpu_count(), buffered_size)


def test_reader(test_list, buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(test_mapper, reader,
                                      cpu_count(), buffered_size)


if __name__ == '__main__':
    for im in train_reader('train.list'):
        print len(im[0])
    for im in train_reader('test.list'):
        print len(im[0])
