import random
from paddle.v2.image import load_and_transform


def train_reader(train_list):
    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            random.shuffle(lines)
            for line in lines:
                img_path, lab = line.strip().split('\t')
                im = load_and_transform(img_path, 256, 224, True)
                yield im.flatten().astype('float32'), int(lab)

    return reader


def test_reader(test_list):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                im = load_and_transform(img_path, 256, 224, False)
                yield im.flatten().astype('float32'), int(lab)

    return reader


if __name__ == '__main__':
    for im in train_reader('train.list'):
        print len(im[0])
    for im in train_reader('test.list'):
        print len(im[0])
