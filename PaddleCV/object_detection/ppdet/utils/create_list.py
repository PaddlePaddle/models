import os
import os.path as osp
import re
import random
import shutil

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
                new_ann_path = osp.join('./VOCdevkit/VOC_all/Annotations/', name_prefix + '.xml') 
                new_img_path = osp.join('./VOCdevkit/VOC_all/JPEGImages/', name_prefix + '.jpg')
                shutil.copy(ann_path, new_ann_path)
                shutil.copy(img_path, new_img_path)
                img_ann_list.append(name_prefix)

    return trainval_list, test_list


def prepare_filelist(devkit_dir, years, output_dir):
    os.makedirs('./VOCdevkit/VOC_all/Annotations/')
    os.makedirs('./VOCdevkit/VOC_all/ImageSets/Main/')
    os.makedirs('./VOCdevkit/VOC_all/JPEGImages/')
    trainval_list = []
    test_list = []
    for year in years:
        trainval, test = walk_dir(devkit_dir, year)
        trainval_list.extend(trainval)
        test_list.extend(test)
    random.shuffle(trainval_list)
    with open(osp.join(output_dir, 'train.txt'), 'w') as ftrainval:
        for item in trainval_list:
            ftrainval.write(item + '\n')
            
    random.shuffle(test_list)
    with open(osp.join(output_dir, 'val.txt'), 'w') as fval:
        with open(osp.join(output_dir, 'test.txt'), 'w') as ftest:
            ct = 0
            for item in test_list:
                ct += 1
                fval.write(item + '\n')
                if ct <= 1000:
                    ftest.write(item + '.jpg' + '\n')
                    
