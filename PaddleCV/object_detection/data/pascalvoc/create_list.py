import os
import os.path as osp
import re
import random

devkit_dir = './VOCdevkit'
years = ['2007', '2012']


def get_dir(devkit_dir, year, type):
    return osp.join(devkit_dir, 'VOC' + year, type)


def walk_dir(devkit_dir, year):
    filelist_dir = get_dir(devkit_dir, year, 'ImageSets/Main')
    annotation_dir = get_dir(devkit_dir, year, 'Annotations')
    img_dir = get_dir(devkit_dir, year, 'JPEGImages')
    trainval_list = []
    test_list = []
    added = set()

    for _, _, files in os.walk(filelist_dir):
        for fname in files:
            img_ann_list = []
            if re.match('[a-z]+_trainval\.txt', fname):
                img_ann_list = trainval_list
            elif re.match('[a-z]+_test\.txt', fname):
                img_ann_list = test_list
            else:
                continue
            fpath = osp.join(filelist_dir, fname)
            for line in open(fpath):
                name_prefix = line.strip().split()[0]
                if name_prefix in added:
                    continue
                added.add(name_prefix)
                ann_path = osp.join(annotation_dir, name_prefix + '.xml')
                img_path = osp.join(img_dir, name_prefix + '.jpg')
                assert os.path.isfile(ann_path), 'file %s not found.' % ann_path
                assert os.path.isfile(img_path), 'file %s not found.' % img_path
                img_ann_list.append((img_path, ann_path))

    return trainval_list, test_list


def prepare_filelist(devkit_dir, years, output_dir):
    trainval_list = []
    test_list = []
    for year in years:
        trainval, test = walk_dir(devkit_dir, year)
        trainval_list.extend(trainval)
        test_list.extend(test)
    random.shuffle(trainval_list)
    with open(osp.join(output_dir, 'trainval.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item[0] + ' ' + item[1] + '\n')

    with open(osp.join(output_dir, 'test.txt'), 'w') as ftest:
        for item in test_list:
            ftest.write(item[0] + ' ' + item[1] + '\n')


if __name__ == '__main__':
    prepare_filelist(devkit_dir, years, '.')
