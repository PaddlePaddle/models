from __future__ import absolute_import
import os
import sys
import cv2
import numpy as np
import shutil
import pdb


def pascal_classes():
    classes = {
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
    return classes


def process_dir(_dir):
    if os.path.exists(_dir):
        shutil.rmtree(_dir)
    os.makedirs(_dir)


def pascal_palette():
    palette = {
        (0, 0, 0): 0,
        (128, 0, 0): 1,
        (0, 128, 0): 2,
        (128, 128, 0): 3,
        (0, 0, 128): 4,
        (128, 0, 128): 5,
        (0, 128, 128): 6,
        (128, 128, 128): 7,
        (64, 0, 0): 8,
        (192, 0, 0): 9,
        (64, 128, 0): 10,
        (192, 128, 0): 11,
        (64, 0, 128): 12,
        (192, 0, 128): 13,
        (64, 128, 128): 14,
        (192, 128, 128): 15,
        (0, 64, 0): 16,
        (128, 64, 0): 17,
        (0, 192, 0): 18,
        (128, 192, 0): 19,
        (0, 64, 128): 20,
        (224, 224, 192): 0
    }
    return palette


def convert_from_color_label(img):
    '''Convert the Pascal VOC label format to train.
    
    Args:
        img: The label result of Pascal VOC.
    '''
    palette = pascal_palette()
    for c, i in palette.items():
        _c = (c[2], c[1], c[0])  # the image channel read by opencv is (b, g, r)
        m = np.all(img == np.array(_c).reshape(1, 1, 3), axis=2)
        img[m] = i
    return img


def main():
    out_dir = 'voc_processed'
    process_dir(out_dir)
    out_train_f = open('voc2012_trainval.txt', 'w')
    out_test_f = open('voc2007_test.txt', 'w')

    devkit_dir = 'VOCdevkit'
    trainval_file = os.path.join(devkit_dir, 'VOC2012', 'ImageSets',
                                 'Segmentation', 'trainval.txt')
    segclass_dir = os.path.join(devkit_dir, 'VOC2012', 'SegmentationClass')
    train_image_dir = os.path.join(devkit_dir, 'VOC2012', 'JPEGImages')
    test_image_dir = os.path.join(devkit_dir, 'VOC2007', 'JPEGImages')
    test_file = os.path.join(devkit_dir, 'VOC2007', 'ImageSets', 'Segmentation',
                             'test.txt')

    with open(trainval_file, 'r') as input_f:
        for line in input_f:
            img = cv2.imread(
                os.path.join(segclass_dir, '%s.png' % line.strip()))
            img = convert_from_color_label(img)

            out_label_path = os.path.join(out_dir, '%s.png' % line.strip())
            cv2.imwrite(out_label_path, img)

            img_path = os.path.join(train_image_dir, '%s.jpg' % line.strip())
            assert (os.path.exists(img_path))

            out_train_f.write('%s %s \n' % (img_path, out_label_path))
            out_train_f.flush()
    out_train_f.close()

    with open(test_file, 'r') as input_f:
        for line in input_f:
            img_path = os.path.join(test_image_dir, '%s.jpg \n' % line.strip())
            out_test_f.write(img_path)
    out_test_f.close()


if __name__ == '__main__':
    main()
